"""Build dashboard/index.html: one offline page (no server, no internet) from committed files only.

  python dashboard/build_dashboard.py                       # results tabs only (no model weights)
  python dashboard/build_dashboard.py --assets dashboard/dashboard_assets.json   # + live detector
  python dashboard/build_dashboard.py --assets ... --check  # rebuild and fail if index.html differs

Inputs (read only): experiments/results/{02_main,04_sweep,06_fu2,10_rf}/..., and optionally the json written by
make_dashboard_assets.py (real deployed models, scaler, sampled held-out test rows, PyTorch reference outputs).
Every number shown in the page comes from these files; the page itself only re-computes the model outputs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments" / "results"
HERE = Path(__file__).resolve().parent


def rcsv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    if v in ("", None):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def clean(rows, keep):
    return [{k: num(r[k]) for k in keep if k in r} for r in rows]


def results():
    d = {}
    d["sweep_decisions"] = clean(rcsv(R / "04_sweep/decisions.csv"), [
        "poison_frac", "seed", "arm", "experience", "attempt", "accepted", "gate_reasons", "Syn_val_recall",
        "Syn_floor", "Syn_val_mean_p", "Syn_ref_mean_p", "benign_far", "new_recall_min", "repair_method",
        "w_flipped", "w_clean"])
    d["sweep_deployed"] = clean(rcsv(R / "04_sweep/deployed_models.csv"), [
        "poison_frac", "seed", "arm", "updates", "r1_ran", "r1_accepted", "classes_learned", "test_benign_far",
        "test_macro_recall", "test_recall_BENIGN", "test_recall_Syn", "test_recall_LDAP", "test_recall_UDP"])
    fu2 = []
    for s in (42, 43, 44):
        for r in json.loads((R / f"06_fu2/fu2/cii/fu2_relabel_f1.0_s{s}/poisoned_relabel/rounds.json").read_text()):
            tw = r.get("teacher_weights") or {}
            v = r["val"]
            fu2.append({"seed": s, "experience": r["experience"], "attempt": r["attempt"], "accepted": r["accepted"],
                        "gate_reasons": "; ".join(r["gate_reasons"]),
                        "Syn_val_recall": v["recall"].get("Syn", 0.0),
                        "Syn_floor": r["ref_val"]["recall"]["Syn"] - 0.05,
                        "Syn_val_mean_p": v["mean_conf"].get("Syn"), "benign_far": v["benign_far"],
                        "new_class_far": v["new_class_far"],
                        "new_recall_min": min([v["recall"].get(k, 0.0) for k in r["new_classes"]] or [None]),
                        "new_classes": r["new_classes"], "n_relabelled": tw.get("n_relabelled"),
                        "n_relabelled_to_true": tw.get("n_relabelled_to_true"),
                        "n_replay_flipped": r["n_replay_flipped"]})
    d["fu2_rounds"] = fu2
    d["fu2_test"] = clean(rcsv(R / "06_fu2/test/test_recall.csv"), [
        "seed", "arm", "BENIGN", "Syn", "LDAP", "UDP", "macro_recall", "benign_far", "n_benign"])
    d["main_attempts"] = clean(rcsv(R / "02_main/report/attempts.csv"), [
        "arm", "experience", "attempt", "accepted", "Syn_recall", "benign_far", "new_class_far", "new_recall_min",
        "replay_flipped"])
    d["main_test"] = {arm: json.loads((R / f"02_main/{arm}/final_test.json").read_text())["test"]["recall"]
                      for arm in ("clean", "poisoned", "poisoned_repair")}
    d["rf_traj"] = clean(rcsv(R / "10_rf/rf/trajectory_val.csv"), [
        "rho", "seed", "arm", "t", "flipped_share", "c1_accepted", "r1_ran", "Syn_recall", "Syn_mean_p"])
    rft = []
    for p in sorted((R / "10_rf/test/rf/test").glob("rf_ref6_r*_s*/*.json")):
        j = json.loads(p.read_text())
        rho, seed = re.match(r"rf_ref6_r([\d.]+)_s(\d+)", p.parent.name).groups()
        last = j["trajectory"][-1]["test"]
        rft.append({"rho": float(rho), "seed": int(seed), "arm": j["arm"], "Syn_recall": last["recall"]["Syn"],
                    "Syn_mean_p": last["mean_conf"]["Syn"], "benign_far": last["benign_far"]})
    d["rf_test"] = rft
    d["followups"] = [x for x in (confgate(), five(), seeds(d)) if x]
    d["funnel"] = funnel()
    d["canon"] = canon_labels()
    return d


def funnel():
    """Data funnel for the Evidence tab, from the committed Stage 1 / 1b manifests."""
    st = json.loads((R / "01_data/split_manifest.json").read_text())["stats"]
    exp = json.loads((R / "01_data/cii/experience_manifest.json").read_text())["counts_per_experience"]
    files = ROOT / "notebooks/manifest.json"
    return {"raw_rows": sum(st["raw_rows_by_label"].values()), "raw_labels": len(st["raw_rows_by_label"]),
            "raw_files": len(json.loads(files.read_text())["files"]) if files.exists() else None,
            "distinct": sum(st["distinct_rows_by_label"].values()), "conflicts": st["n_conflict_vectors"],
            "kept": sum(st["kept_by_label"].values()),
            "cap": max(v for k, v in st["sampled_before_conflict_drop"].items() if k != "BENIGN"),
            "train_4class": sum(sum(v.values()) for v in exp.values())}


def canon_labels():
    """Label spellings -> canonical names, read from common.py's _CANON table (no import, so no pandas needed)."""
    tree = ast.parse((ROOT / "common.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_CANON" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("common.py: _CANON not found")


SAMPLE_PER_CLASS = 50


def sample_csv(assets):
    """200 held-out test rows (the first 50 per class of the bundled sample), raw values, with a Label column.
    Embedded in the page ("Load sample") and written to dashboard/sample_flows.csv."""
    rows, feats = assets["rows"], assets["scaler"]["features_numeric"]
    pick = []
    for c in ("BENIGN", "Syn", "LDAP", "UDP"):
        pick += [i for i, l in enumerate(rows["label"]) if l == c][:SAMPLE_PER_CLASS]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["row_id"] + feats + ["Protocol", "Label"])
    for i in pick:
        w.writerow([rows["rid"][i]] + [repr(float(v)) for v in rows["num"][i]] + [int(rows["cat"][i][0]), rows["label"][i]])
    return buf.getvalue()


SCHED4, SCHED5 = ("BENIGN", "Syn", "LDAP", "UDP"), ("BENIGN", "Syn", "LDAP", "UDP", "TFTP")


def seeds(d):
    """SEEDS follow-up: seeds 45-49 (13_seeds/summary.json, test.json), pooled with the committed seeds 42-44
    (04_sweep/deployed_models.csv, 06_fu2/test/test_recall.csv, 12_five/test.json) for the Results chart and KPIs."""
    base = R / "13_seeds"
    sp, tp = base / "summary.json", base / "test.json"
    if not sp.exists():
        return None
    summ = json.loads(sp.read_text())
    test = {r["name"]: r for r in json.loads(tp.read_text())["runs"]} if tp.exists() else {}
    title, tag = "Follow-up: five more seeds (45-49)", "4-class soft: registered; relabel and 5-class: post hoc"
    groups = [("4 classes, f 0.75-0.95, soft repair", 4, (0.75, 0.9, 0.95), "poisoned_repair"),
              ("4 classes, f 1.0, relabel repair", 4, (1.0,), "poisoned_relabel"),
              ("5 classes, f 0.9, soft repair", 5, (0.9,), "poisoned_repair"),
              ("5 classes, f 1.0, relabel repair", 5, (1.0,), "poisoned_relabel")]
    arms = (("poisoned", "Gate only"), ("poisoned_repair", "Gate + repair"), ("clean", "Clean"))

    def new_runs(k, fracs):
        return [e for e in summ["runs"] if e["classes"] == k and any(abs(e["frac"] - f) < 1e-9 for f in fracs)]

    def learned(rec, cls):
        return all((rec.get(c) or 0) > 0 for c in cls)

    def old_learned(k, fracs, arm):  # seeds 42-44, committed test results
        if k == 4 and fracs == (1.0,) and arm == "poisoned_repair":
            return [learned({c: r[c] for c in SCHED4}, SCHED4) for r in d["fu2_test"] if r["arm"] == "poisoned_relabel"]
        if k == 4:
            return [learned({c: r[f"test_recall_{c}"] for c in SCHED4}, SCHED4) for r in d["sweep_deployed"]
                    if r["arm"] == arm and any(abs(r["poison_frac"] - f) < 1e-9 for f in fracs)]
        five_t = json.loads((R / "12_five/test.json").read_text())["runs"]
        return [learned(r[arm]["recall"], SCHED5) for r in five_t if any(abs(r["frac"] - f) < 1e-9 for f in fracs)]

    rows, chart, pooled = [], [], {"seeds": "42-49"}
    for label, k, fracs, rep_arm in groups:
        cls = SCHED4 if k == 4 else SCHED5
        runs = new_runs(k, fracs)
        bars = []
        for arm, nm in arms:
            acc = sum(e[arm]["updates_accepted"] for e in runs)
            tot = sum(e[arm]["updates"] for e in runs)
            r1 = f'{sum(e[arm]["r1_accepted"] for e in runs)}/{sum(e[arm]["r1_ran"] for e in runs)}'
            t = [test[e["name"]][arm] for e in runs if e["name"] in test]
            cells = ["", "", "", "", ""]
            if t:
                new_k = sum(learned(x["recall"], cls) for x in t)
                old = old_learned(k, fracs, arm)
                pk, pn = new_k + sum(old), len(t) + len(old)
                cells = [f"{new_k}/{len(t)}", f"{pk}/{pn}", f"{sum(x['accuracy'] for x in t) / len(t):.4f}",
                         f"{sum(x['macro_recall'] for x in t) / len(t):.4f}", f"{max(x['benign_far'] for x in t):.4f}"]
                bars.append({"arm": rep_arm if arm == "poisoned_repair" else arm, "k": pk, "n": pn})
                if k == 4 and arm in ("poisoned", "poisoned_repair"):
                    key = "soft" if rep_arm == "poisoned_repair" else "relabel"
                    pooled.setdefault(key, {})["repair" if arm == "poisoned_repair" else "gate"] = [pk, pn]
            name = nm if arm != "poisoned_repair" else ("Gate + soft repair" if rep_arm == "poisoned_repair" else "Gate + relabel repair")
            rows.append(f"<tr><td>{label if arm == 'poisoned' else ''}</td><td>{name}</td><td class='num'>{acc}/{tot}</td>"
                        f"<td class='num'>{r1}</td>" + "".join(f"<td class='num'>{c}</td>" for c in cells) + "</tr>")
        if bars:
            chart.append({"label": label.replace(", soft repair", "").replace(", relabel repair", "").replace("classes", "cl."), "bars": bars})
    html = ("<table class='t'><tr><th>Setting</th><th>Arm</th><th>Updates accepted</th><th>R1 passed/ran</th>"
            "<th>All classes learned, seeds 45-49</th><th>Pooled 42-49</th><th>Accuracy</th><th>Macro recall</th>"
            "<th>Max benign FAR</th></tr>" + "".join(rows) + "</table><div class='note'>Accuracy, macro recall and benign FAR: "
            "held-out test, seeds 45-49 (mean; FAR = maximum). Pooled counts add the committed seeds 42-44 (sweep, FU2, FIVE) "
            "and are descriptive. Pre-registration: experiments/prereg_seeds.md. "
            + ("Test scored once after the decisions were committed." if test else "Validation decisions only; test not yet scored.")
            + "</div>")
    out = {"id": "seeds", "title": title, "tag": tag, "html": html}
    if test and "soft" in pooled and "relabel" in pooled:
        s_, r_ = pooled["soft"], pooled["relabel"]
        out.update({"pooled": pooled, "chart": chart,
                    "takeaway": (f"Over seeds 42-49, soft repair learns all four classes in {s_['repair'][0]}/{s_['repair'][1]} runs at "
                                 f"f 0.75-0.95 (gate only {s_['gate'][0]}/{s_['gate'][1]}); relabel repair in {r_['repair'][0]}/"
                                 f"{r_['repair'][1]} at f 1.0 (gate only {r_['gate'][0]}/{r_['gate'][1]})."),
                    "more": "Bars: runs whose deployed model has held-out test recall above 0 on every class of its schedule, "
                            "pooled over seeds 42-44 (committed earlier) and 45-49 (SEEDS). Descriptive; the registered "
                            "predictions S1-S4 concern seeds 45-49 only."})
    return out


def five():
    """FIVE follow-up (post hoc): fifth class TFTP. pilot.json / summary.json (validation) and test.json."""
    base = R / "12_five"
    pp, sp, tp = base / "pilot.json", base / "summary.json", base / "test.json"
    title = "Follow-up: a fifth class (E3 adds TFTP)"
    if sp.exists():
        summ = json.loads(sp.read_text())
    elif pp.exists():
        pil = json.loads(pp.read_text())
        if not pil["proceed"]:
            bad = [f"seed {s}: " + "; ".join("; ".join(x["reasons"]) for x in v if not x["accepted"])
                   for s, v in pil["per_seed"].items() if not all(x["accepted"] for x in v)]
            return {"id": "five", "title": title, "tag": "post hoc, stopped", "stats": [],
                    "html": "<div>The registered clean pilot did not pass, so no poisoned run was made. Rejected clean "
                            "updates: " + " | ".join(bad) + "</div>"}
        return None
    else:
        return None
    test = {r["name"]: r for r in json.loads(tp.read_text())["runs"]} if tp.exists() else {}
    cls = ["BENIGN", "Syn", "LDAP", "UDP", "TFTP"]
    rows, stats = [], []
    for f, label in ((0.9, "90% flipped, soft repair"), (1.0, "100% flipped, relabel repair")):
        runs = [e for e in summ["runs"] if abs(e["frac"] - f) < 1e-9]
        for arm, nm in (("clean", "Clean"), ("poisoned", "Gate only"), ("poisoned_repair", "Gate + repair")):
            r1 = f'{sum(e[arm]["r1_accepted"] for e in runs)}/{sum(e[arm]["r1_ran"] for e in runs)}'
            t = [test[e["name"]][arm] for e in runs if e["name"] in test]
            learned = f'{sum(all(x["recall"].get(k, 0) > 0 for k in cls) for x in t)}/{len(t)}' if t else ""
            per = "".join(f"<td class='num'>{sum(x['recall'].get(k, 0) for x in t) / len(t):.4f}</td>" if t else
                          "<td></td>" for k in cls)
            far = f"{sum(x['benign_far'] for x in t) / len(t):.4f}" if t else ""
            stats.append({"frac": f, "arm": arm, "r1_accepted": sum(e[arm]["r1_accepted"] for e in runs),
                          "r1_ran": sum(e[arm]["r1_ran"] for e in runs), "n_test": len(t),
                          "all5": sum(all(x["recall"].get(k, 0) > 0 for k in cls) for x in t) if t else None,
                          "benign_far": [x["benign_far"] for x in t]})
            rows.append(f"<tr><td>{label if arm == 'clean' else ''}</td><td>{nm}</td><td class='num'>{r1}</td>"
                        f"<td class='num'>{learned}</td>{per}<td class='num'>{far}</td></tr>")
    html = ("<table class='t'><tr><th>Setting</th><th>Arm</th><th>R1 passed/ran</th><th>All 5 learned</th>"
            + "".join(f"<th>{k}</th>" for k in cls) + "<th>Benign FAR</th></tr>" + "".join(rows) + "</table>"
            "<div class='note'>Recall columns: held-out test, mean of 3 seeds. Pre-registration: "
            "experiments/prereg_five.md. " + ("Test scored once after the decisions were committed." if test else
                                               "Validation decisions only; test not yet scored.") + "</div>")
    return {"id": "five", "title": title, "tag": "post hoc, 2026-11-19", "html": html, "stats": stats}


def confgate():
    """CG follow-up (post hoc): summary.json (validation decisions) and, once scored, test.json."""
    sp, tp = R / "11_confgate" / "summary.json", R / "11_confgate" / "test.json"
    if not sp.exists():
        return None
    summ = json.loads(sp.read_text())
    test = {r["name"]: r for r in json.loads(tp.read_text())["runs"]} if tp.exists() else {}
    arms = [("clean", "Clean"), ("poisoned", "Gate only"), ("poisoned_repair", "Gate + soft repair")]
    groups = [("CG-RF", 0.1, "Refresh, 10% more flips per update"), ("CG-RF", 0.05, "Refresh, 5% more flips per update"),
              ("CG-MAIN", 0.5, "Four classes, 50% flipped")]
    rows, stats = [], []
    for setting, frac, label in groups:
        runs = [e for e in summ["runs"] if e["setting"] == setting and abs(e["frac"] - frac) < 1e-9]
        if not runs:
            continue
        for arm, name in arms:
            acc = sum(e[arm]["updates_accepted"] for e in runs)
            tot = sum(e[arm]["updates"] for e in runs)
            r1 = f'{sum(e[arm]["r1_accepted"] for e in runs)}/{sum(e[arm]["r1_ran"] for e in runs)}'
            t = [test[e["name"]][arm] for e in runs if e["name"] in test]
            conf = sum(x["mean_conf"].get("Syn", 0) for x in t) / len(t) if t else None
            rec = sum(x["recall"].get("Syn", 0) for x in t) / len(t) if t else None
            learned = None
            if setting == "CG-MAIN" and t:
                learned = sum(all(x["recall"].get(k, 0) > 0 for k in ("BENIGN", "Syn", "LDAP", "UDP")) for x in t)
            stats.append({"setting": setting, "frac": frac, "arm": arm, "updates_accepted": acc, "updates": tot,
                          "n_test": len(t), "Syn_recall": rec, "Syn_mean_conf": conf})
            rows.append(f"<tr><td>{label if arm == 'clean' else ''}</td><td>{name}</td><td class='num'>{acc}/{tot}</td>"
                        f"<td class='num'>{r1}</td><td class='num'>{'' if rec is None else f'{rec:.4f}'}</td>"
                        f"<td class='num'>{'' if conf is None else f'{conf:.4f}'}</td>"
                        f"<td class='num'>{'' if learned is None else f'{learned}/{len(t)}'}</td></tr>")
    html = ("<table class='t'><tr><th>Setting</th><th>Arm</th><th>Updates accepted</th><th>R1 passed/ran</th>"
            "<th>Syn recall (test)</th><th>Syn confidence (test)</th><th>4 classes learned</th></tr>" + "".join(rows)
            + "</table><div class='note'>Confidence gate: reject if an old attack class's mean confidence on "
            f"validation falls more than {summ['conf_drop']} below the last accepted model. Threshold fixed from clean "
            "logs before the run (experiments/prereg_confgate.md). 3 seeds per row; test columns are means. "
            + ("Held-out test scored once after the decisions were committed." if test else
               "Validation decisions only; test not yet scored.") + "</div>")
    return {"id": "confgate", "title": "Follow-up: a confidence-aware gate", "tag": "post hoc, 2026-11-19", "html": html,
            "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=None, help="json from make_dashboard_assets.py (optional)")
    ap.add_argument("--out", default=str(HERE / "index.html"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = {"results": results(), "assets": None, "sample_csv": None}
    if a.assets:
        data["assets"] = json.loads(Path(a.assets).read_text())
        data["sample_csv"] = sample_csv(data["assets"])
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "template.html").read_text()
    page = page.replace("/*__ENGINE__*/", (HERE / "engine.js").read_text())
    page = page.replace("__DATA__", blob)
    out = Path(a.out)
    sample = HERE / "sample_flows.csv"
    if a.check:
        same = out.exists() and out.read_text() == page
        if data["sample_csv"] is not None:
            same = same and sample.exists() and sample.read_text() == data["sample_csv"]
        print("dashboard/index.html is up to date" if same else "dashboard/index.html (or sample_flows.csv) differs from a fresh build")
        sys.exit(0 if same else 1)
    out.write_text(page)
    if data["sample_csv"] is not None:
        sample.write_text(data["sample_csv"])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) | live models: {'yes' if a.assets else 'no'}")


if __name__ == "__main__":
    main()
