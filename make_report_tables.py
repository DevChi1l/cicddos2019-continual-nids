"""Report tables from COMMITTED result files only (reads git-tracked JSON/CSV; trains, scores and downloads nothing).

  python make_report_tables.py            # (re)write docs/RESULT_TABLES.md
  python make_report_tables.py --check    # rebuild in memory; exit 1 if docs/RESULT_TABLES.md differs, or if any
                                          # recomputed aggregate disagrees with its committed summary file

Every table names its source files with their sha256 (first 12 hex digits), so a changed source changes the
document and `--check` fails. Before writing, the script recomputes aggregates from per-seed files and asserts that
they equal the committed summary CSVs, and that experiments meant to share test rows have one row hash:
  sweep  04_sweep/test_summary.csv          == mean/min/max of 04_sweep/final_test.csv
  FU2    06_fu2/test/test_summary.csv       == mean/min/max of 06_fu2/test/test_recall.csv
  oracle 07_oracle/test/test_median.csv     == median of 07_oracle/test/test_recall.csv (CII)
  RF     10_rf/test/test_summary.csv (A6)   == mean/min/max of 10_rf/test/test_final.csv
  four-label test rows (main, sweep, FU2) one sha256; 12-label test rows (all12, FU1, oracle) one sha256
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "experiments" / "results"
OUT = HERE / "docs" / "RESULT_TABLES.md"
TOL = 1e-12


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def src(*rels: str) -> str:
    return "Source: " + ", ".join(f"`{r}` ({sha(HERE / r)})" for r in rels)


def jl(rel: str):
    return json.loads((HERE / rel).read_text())


def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(HERE / rel)


def f4(x) -> str:
    return "-" if pd.isna(x) else f"{x:.4f}"


def rng(s: pd.Series) -> str:
    lo, hi = s.min(), s.max()
    return f4(lo) if abs(hi - lo) < 5e-5 else f"{f4(lo)}-{f4(hi)}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ---- consistency checks (recompute aggregates from per-seed files) ---------------------------------
def check_close(a: pd.Series, b: pd.Series, what: str):
    """a (recomputed) and b (committed) must have the same keys and values."""
    assert sorted(a.index) == sorted(b.index), f"{what}: different groups {sorted(a.index)} vs {sorted(b.index)}"
    d = (a - b.reindex(a.index)).abs().max()
    assert d < TOL, f"{what}: recomputed aggregate differs from the committed summary by {d}"


def consistency() -> list[str]:
    notes = []
    ft, ts = csv("experiments/results/04_sweep/final_test.csv"), csv("experiments/results/04_sweep/test_summary.csv")
    ft = ft[ft["run"].str.startswith("sweep_")]  # repro_s42 (f = 0.5, the H0 check) is not in the summary
    for k in ("Syn", "LDAP", "UDP", "BENIGN"):
        g = ft.groupby(["poison_frac", "arm"])[f"recall_{k}"]
        for s in ("mean", "min", "max"):
            check_close(getattr(g, s)().sort_index(), ts.set_index(["poison_frac", "arm"])[f"recall_{k}_{s}"].sort_index(),
                        f"sweep {k} {s}")
    notes.append("sweep: `test_summary.csv` = mean/min/max of the 36 `sweep_*` rows of `final_test.csv`")

    tr, tsum = csv("experiments/results/06_fu2/test/test_recall.csv"), csv("experiments/results/06_fu2/test/test_summary.csv")
    for k in ("BENIGN", "Syn", "LDAP", "UDP", "macro_recall", "benign_far"):
        g = tr.groupby("arm")[k]
        for s in ("mean", "min", "max"):
            check_close(getattr(g, s)().sort_index(), tsum.set_index("arm")[f"{k}_{s}"].sort_index(), f"FU2 {k} {s}")
    notes.append("FU2: `test_summary.csv` = mean/min/max of `test_recall.csv` (4 arms x 6 metrics)")

    orc, med = csv("experiments/results/07_oracle/test/test_recall.csv"), csv("experiments/results/07_oracle/test/test_median.csv")
    m = orc[orc["scenario"] == "cii"].groupby("setting").median(numeric_only=True)
    mm = med[med["scenario"].str.startswith("cii")].set_index("setting")
    for k in ("Syn", "DNS", "LDAP", "SSDP", "UDP", "macro_recall", "benign_far"):
        check_close(m[k].sort_index(), mm[k].sort_index(), f"oracle {k}")
    notes.append("oracle: CII rows of `test_median.csv` = median over seeds 42-44 of `test_recall.csv`")

    rf, rs = csv("experiments/results/10_rf/test/test_final.csv"), csv("experiments/results/10_rf/test/test_summary.csv")
    a6 = rs[rs["model"] == "A6"].set_index(["rho", "arm"]).sort_index()
    for k in ("Syn_recall", "Syn_mean_p", "benign_far", "TFTP_recall"):
        g = rf.groupby(["rho", "arm"])[k]
        for s in ("mean", "min", "max"):
            check_close(getattr(g, s)().sort_index(), a6[f"{k}_{s}"], f"RF {k} {s}")
    notes.append("RF: A6 rows of `test_summary.csv` = mean/min/max of `test_final.csv`")

    four = {*csv("experiments/results/04_sweep/final_test.csv")["test_rows_sha256"],
            *csv("experiments/results/06_fu2/test/test_recall.csv")["test_rows_sha256"],
            *csv("experiments/results/02_main/report/final_test.csv")["test_rows_sha256"]}
    assert len(four) == 1, f"four-label test rows differ: {four}"
    twelve = {*csv("experiments/results/04b_all12/final_test.csv")["test_rows_sha256"],
              *csv("experiments/results/05_fu1/test/test_summary.csv")["test_rows_sha256"]}
    for p in sorted((R / "07_oracle" / "test" / "oracle").rglob("test.json")):
        twelve.add(json.loads(p.read_text())["test"]["rows_sha256"])
    assert len(twelve) == 1, f"12-label test rows differ: {twelve}"
    notes.append(f"four-label test rows: one sha256 `{next(iter(four))[:12]}` across main, sweep and FU2; "
                 f"12-label test rows: one sha256 `{next(iter(twelve))[:12]}` across all12, FU1 and the oracle")
    return notes


# ---- tables -----------------------------------------------------------------------------------------
def t_data() -> str:
    a, s = jl("notebooks/manifest.json"), jl("experiments/results/01_data/split_manifest.json")
    st, cps = s["stats"], s["counts_per_split"]
    labels = sorted(a["label_totals"], key=lambda k: (k != "BENIGN", -a["label_totals"][k]))
    rows = []
    for k in labels:
        rows.append({"label": k, "raw rows": f"{a['label_totals'][k]:,}",
                     "non-finite rows": f"{st['nonfinite_rows_by_label'].get(k, 0):,}",
                     "distinct finite": f"{st['distinct_rows_by_label'][k]:,}",
                     "in a label conflict": f"{st['conflict_distinct_by_label'].get(k, 0):,}",
                     "kept": f"{st['kept_by_label'][k]:,}",
                     **{sp: f"{cps[sp].get(k, 0):,}" for sp in ("train", "calib", "val", "test")}})
    tot = sum(a["label_totals"].values())
    head = (f"{len(a['files'])} CSV files, {tot:,} raw rows, {sum(f['bytes'] for f in a['files']) / 1e9:.2f} GB, "
            f"{len(a['label_totals'])} raw labels, {sum(f['exact_duplicate_rows'] for f in a['files']):,} within-file "
            f"exact duplicate rows. Stage 1: cap {s['args']['cap']:,} distinct rows per attack label (BENIGN never "
            f"capped), non-finite policy `{s['args']['nonfinite']}`, split `{s['args']['split']}` "
            f"{'/'.join(str(int(v * 100)) for v in s['split_fracs'].values())}, seed {s['seed']}, "
            f"{st['n_conflict_vectors']:,} label-conflict vectors dropped, cross-split identical inputs "
            f"{s['cross_split_feature_overlap']}.")
    return ("## T1. Raw audit to Stage 1 splits (all 14 raw labels)\n\n" + head + "\n\n" + md(pd.DataFrame(rows)) +
            "\n\n" + src("notebooks/manifest.json", "experiments/results/01_data/split_manifest.json"))


def t_files() -> str:
    a = jl("notebooks/manifest.json")
    rows = []
    for f in a["files"]:
        name = f["path"].split("csv-files/")[-1]
        lab = sorted(f["label_counts"].items(), key=lambda kv: -kv[1])
        rows.append({"day/file": name, "rows": f"{f['rows_scanned']:,}",
                     "labels found (label column)": ", ".join(f"{k} {v:,}" for k, v in lab),
                     "exact duplicate rows": f"{f['exact_duplicate_rows']:,}"})
    return ("## T2. Source files: the label column, not the filename, decides the class\n\n" + md(pd.DataFrame(rows)) +
            "\n\n" + src("notebooks/manifest.json"))


def t_subsets() -> str:
    four = jl("experiments/results/01_data/cii/experience_manifest.json")
    rf = jl("experiments/results/10_rf/ref6/cii/experience_manifest.json")
    a12 = jl("experiments/results/04b_all12/schedule.json")
    rows = []
    for name, m in (("four-label (main, sweep, FU2)", four), ("three-label RF (`ref6`)", rf)):
        for e, cnt in m["counts_per_experience"].items():
            rows.append({"subset": name, "experience": e,
                         "train rows": ", ".join(f"{k} {v:,}" for k, v in cnt.items())})
    for i, e in enumerate(a12["experiences"]):
        rows.append({"subset": "12-label all12 (CII)", "experience": f"E{i}",
                     "train rows": ", ".join([f"BENIGN {a12['cii_benign_per_experience'][i]:,}"] +
                                             [f"{k} {a12['attack_cap']:,}" for k in e])})
    val = pd.DataFrame({"four-label": four["protected_exclusions"]["kept_by_split_class"]["val"],
                        "RF": rf["protected_exclusions"]["kept_by_split_class"]["val"]}).fillna(0).astype(int)
    tst = pd.DataFrame({"four-label": four["protected_exclusions"]["kept_by_split_class"]["test"],
                        "RF": rf["protected_exclusions"]["kept_by_split_class"]["test"]}).fillna(0).astype(int)
    pv = pd.concat({"val": val, "test": tst}, axis=1).map(lambda v: f"{v:,}").reset_index().rename(columns={"index": "class"})
    pv.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in pv.columns]
    return ("## T3. Experimental subsets (rows per experience; all from the same Stage 1 splits)\n\n" +
            f"Four-label: attack cap {four['attack_cap']:,}, {four['protected_exclusions']['n_excluded']} protected rows "
            f"excluded. 12-label: attack cap {a12['attack_cap']:,}, {a12['labels_used']} of {a12['labels_total']} labels "
            f"(excluded: {', '.join(a12['excluded'])}). RF: attack cap {rf['attack_cap']:,}, TFTP split into 7 disjoint "
            f"portions, {rf['protected_exclusions']['n_excluded']} protected rows excluded.\n\n" + md(pd.DataFrame(rows)) +
            "\n\nProtected rows scored (after transformed-duplicate exclusion):\n\n" + md(pv) + "\n\n" +
            src("experiments/results/01_data/cii/experience_manifest.json", "experiments/results/04b_all12/schedule.json",
                "experiments/results/10_rf/ref6/cii/experience_manifest.json"))


def t_main() -> str:
    at, ft = csv("experiments/results/02_main/report/attempts.csv"), csv("experiments/results/02_main/report/final_test.csv")
    d = at[["arm", "experience", "attempt", "accepted", "Syn_recall", "new_recall_min", "benign_far", "replay_flipped"]].copy()
    for c in ("Syn_recall", "new_recall_min", "benign_far"):
        d[c] = d[c].map(f4)
    t = ft[["arm", "recall_Syn", "recall_LDAP", "recall_UDP", "benign_far", "n_test_rows"]].copy()
    for c in ("recall_Syn", "recall_LDAP", "recall_UDP", "benign_far"):
        t[c] = t[c].map(lambda x: f"{x:.5f}")
    return ("## T4. `main` (notebook 02): CII, seed 42, f = 0.5 once before E1\n\nValidation decisions:\n\n" + md(d) +
            "\n\nHeld-out test (scored inside the run; no decision used it):\n\n" + md(t) + "\n\n" +
            src("experiments/results/02_main/report/attempts.csv", "experiments/results/02_main/report/final_test.csv"))


def t_sweep() -> str:
    h = csv("experiments/results/04_sweep/hypotheses.csv")
    dm = csv("experiments/results/04_sweep/deployed_models.csv")
    ts = csv("experiments/results/04_sweep/test_summary.csv")
    dm["all4"] = dm["classes_learned"].str.split().str.len() == 4
    learned = dm.groupby(["poison_frac", "arm"])["all4"].agg(lambda s: f"{int(s.sum())}/{len(s)}")
    rows = []
    for r in ts.itertuples(index=False):
        rows.append({"f": r.poison_frac, "arm": r.arm, "learned all 4": learned[(r.poison_frac, r.arm)],
                     "Syn": f"{f4(r.recall_Syn_mean)} ({f4(r.recall_Syn_min)}-{f4(r.recall_Syn_max)})",
                     "LDAP": f"{f4(r.recall_LDAP_mean)} ({f4(r.recall_LDAP_min)}-{f4(r.recall_LDAP_max)})",
                     "UDP": f"{f4(r.recall_UDP_mean)} ({f4(r.recall_UDP_min)}-{f4(r.recall_UDP_max)})",
                     "benign FAR": f"{f4(r.benign_far_mean)} ({f4(r.benign_far_min)}-{f4(r.benign_far_max)})"})
    return ("## T5. Pre-registered sweep (notebook 04): CII, seeds 42-44, f once before E1\n\nRegistered hypotheses "
            "(validation):\n\n" + md(h) + "\n\nHeld-out test, deployed model after E2, mean (min-max) over 3 seeds:\n\n" +
            md(pd.DataFrame(rows)) + "\n\n" + src("experiments/results/04_sweep/hypotheses.csv",
                                                  "experiments/results/04_sweep/deployed_models.csv",
                                                  "experiments/results/04_sweep/test_summary.csv"))


def t_fu2() -> str:
    tr = csv("experiments/results/06_fu2/test/test_recall.csv")
    order = ["clean", "poisoned", "poisoned_repair", "poisoned_relabel"]
    label = {"clean": "clean (sweep f = 1.0 control)", "poisoned": "poisoned, gate only (sweep)",
             "poisoned_repair": "poisoned + registered soft R1 (sweep)",
             "poisoned_relabel": "**poisoned + teacher relabel R1 (FU2, post hoc)**"}
    rows = []
    for arm in order:
        g = tr[tr["arm"] == arm]
        learned = int(((g[["Syn", "LDAP", "UDP"]] > 0).all(axis=1)).sum())
        rows.append({"arm": label[arm], "seeds with all 4 learned": f"{learned}/{len(g)}",
                     **{k: f"{f4(g[k].mean())} ({rng(g[k])})" for k in ("Syn", "LDAP", "UDP", "macro_recall", "benign_far")}})
    h3 = csv("experiments/results/06_fu2/test/h3.csv")
    return ("## T6. PRIMARY RESULT. Four-label defence at f = 1.0 (FU2, notebook 06; post hoc method)\n\n"
            "Every stored Syn label flipped to BENIGN once before E1 (flips persist). Held-out test, mean (min-max) over "
            f"seeds 42-44, {int(tr['test_rows'].iloc[0]):,} test rows ({int(tr['n_benign'].iloc[0]):,} benign). "
            f"H3 (Syn >= 0.95, LDAP and UDP >= 0.80, benign FAR <= 0.05) met in {int(h3['met'].sum())}/{len(h3)} seeds.\n\n" +
            md(pd.DataFrame(rows)) + "\n\n" + src("experiments/results/06_fu2/test/test_recall.csv",
                                                  "experiments/results/06_fu2/test/h3.csv"))


def t_twelve() -> str:
    ft = csv("experiments/results/04b_all12/final_test.csv")
    fs = csv("experiments/results/05_fu1/test/test_summary.csv")
    rows = []
    for r in ft.itertuples(index=False):
        rows.append({"model": f"s3 {r.scenario} {r.arm}", "labels with recall > 0": len(r.classes_learned.split()),
                     "learned": r.classes_learned, "macro recall (12)": f4(r.test_macro_recall),
                     "benign FAR": f4(r.test_benign_far)})
    for r in fs[fs["model"] == "fu1"].itertuples(index=False):
        rows.append({"model": f"FU1 {r.scenario}", "labels with recall > 0": r.labels_learned, "learned": r.classes,
                     "macro recall (12)": f4(r.macro_recall_12), "benign FAR": f4(r.benign_far)})
    return ("## T7. 12-label continual runs (notebook 04 part B, one seed) and FU1 retries (notebook 05)\n\n" +
            md(pd.DataFrame(rows)) + "\n\n" + src("experiments/results/04b_all12/final_test.csv",
                                                  "experiments/results/05_fu1/test/test_summary.csv"))


def t_oracle() -> str:
    v = csv("experiments/results/07_oracle/test/test_vs_continual.csv")
    v = v[v["scenario"] == "cii"].drop(columns="scenario")
    for c in v.columns[1:]:
        v[c] = v[c].map(f4)
    return ("## T8. Joint oracle (diagnostic, notebook 07) vs continual models, CII test\n\nOracle = one model trained on "
            "all 12 labels at once (median of seeds 42-44; e10 / e20 epochs). Continual = the s3 clean arm (T7) and FU1.\n\n" +
            md(v) + "\n\n" + src("experiments/results/07_oracle/test/test_vs_continual.csv"))


def t_pilots() -> str:
    rows = []
    p = jl("experiments/results/08_rc/rc/pilot.json")
    for x in p["attempts"]:
        rows.append({"experiment": "RC (notebook 08)", "schedule": ", ".join(x["classes"]),
                     "seeds accepting every update": f"{x['seeds_accepting_all']}/3",
                     "rejections per class": ", ".join(f"{k} {n}" for k, n in x["rejections_per_class"].items() if n),
                     "decision": f"proceed={p['proceed']} ({p['reason']})"})
    s = jl("experiments/results/09_rcs/rcs/screen.json")
    for tag, c in s["candidates"].items():
        rej = {}
        for v in c["per_seed"].values():
            for u in v["updates"]:
                if not u["accepted"]:
                    rej[u["new_class"]] = rej.get(u["new_class"], 0) + 1
        rows.append({"experiment": f"RCS `{tag}` (notebook 09)", "schedule": ", ".join(c["classes"]),
                     "seeds accepting every update": f"{sum(v['all_accepted'] for v in c['per_seed'].values())}/3",
                     "rejections per class": ", ".join(f"{k} {n}" for k, n in rej.items()),
                     "decision": f"stable {c['stable_seeds']}/3; selected={s['selected']}"})
    f = jl("experiments/results/10_rf/rf/pilot.json")
    rows.append({"experiment": "RF `ref6` (notebook 10)", "schedule": "6 x refresh (BENIGN + TFTP)",
                 "seeds accepting every update": f"{sum(v['all_accepted'] for v in f['per_seed'].values())}/3",
                 "rejections per class": "none", "decision": f"proceed={f['proceed']} ({f['reason']})"})
    return ("## T9. Clean (unpoisoned) pilots of the repeated-corruption studies, validation only\n\n" +
            md(pd.DataFrame(rows)) + "\n\n" + src("experiments/results/08_rc/rc/pilot.json",
                                                  "experiments/results/09_rcs/rcs/screen.json",
                                                  "experiments/results/10_rf/rf/pilot.json"))


def t_rf() -> str:
    s = csv("experiments/results/10_rf/test/test_summary.csv")
    s = s[s["model"] == "A6"]
    rows = []
    for r in s.itertuples(index=False):
        rows.append({"rho per update": r.rho, "arm": r.arm, "stored Syn flipped": f"{r.flipped_share:.0%}",
                     "Syn recall": f"{f4(r.Syn_recall_mean)} ({f4(r.Syn_recall_min)}-{f4(r.Syn_recall_max)})",
                     "Syn mean p": f"{f4(r.Syn_mean_p_mean)} ({f4(r.Syn_mean_p_min)}-{f4(r.Syn_mean_p_max)})",
                     "TFTP recall": f4(r.TFTP_recall_mean), "benign FAR": f4(r.benign_far_mean)})
    pr = jl("experiments/results/10_rf/rf/summary.json")["predictions_val"]["rho_0.10"]
    pred = pd.DataFrame([
        {"prediction": "RF1 recall ratchet", "registered test (val)": "C1 accepted >= 5/6 in >= 2/3 seeds AND drift > 0.05 in >= 2/3",
         "result": f"accepted {pr['RC1_ratchet']['C1_accepted_at_>=T-1_of_T_seeds']}/3, drift {pr['RC1_ratchet']['drift_>0.05_seeds']}/3 -> "
                   f"{'met' if pr['RC1_ratchet']['met'] else 'NOT supported'}"},
        {"prediction": "RF2 silent confidence loss", "registered test (val)": "Spearman(mean p, flips) < 0 and mean p A6 < A0 in 3/3",
         "result": f"{pr['RC2_confidence_loss']['seeds']}/3 -> {'supported' if pr['RC2_confidence_loss']['met'] else 'not supported'}"},
        {"prediction": "RF3 repair is gated", "registered test (val)": "repair arm = poisoned arm until its first C1 rejection",
         "result": f"{'supported' if pr['RC3_repair_is_gated']['met'] else 'not supported'} (no C1 rejection in "
                   f"{pr['RC3_repair_is_gated']['runs_without_any_C1_rejection']}/3 runs)"}])
    return ("## T10. LIMITATION. RF (notebook 10; post hoc, fixed-class traffic refresh, three labels)\n\nE0 = BENIGN + "
            "Syn + TFTP, six refresh updates, no new class. Held-out test of the deployed model after E6 (A6), mean "
            "(min-max) over seeds 42-44. `poisoned_repair` is identical to `poisoned` (no rejection, no repair deployed).\n\n" +
            md(pd.DataFrame(rows)) + "\n\nRegistered predictions at rho = 0.10 (validation):\n\n" + md(pred) + "\n\n" +
            src("experiments/results/10_rf/test/test_summary.csv", "experiments/results/10_rf/rf/summary.json"))


def build() -> str:
    notes = consistency()
    parts = ["# Result tables (generated; do not edit by hand)",
             "Generated by `python make_report_tables.py` from committed files only. Verify with "
             "`python make_report_tables.py --check` (exit 1 if this file differs from a fresh build or a recomputed "
             "aggregate disagrees with its summary file). Interpretation: `docs/PROJECT_TECHNICAL_HANDBOOK.md` and "
             "`experiments/results/SUMMARY.md`.",
             "Consistency checks passed while generating this file:\n\n" + "\n".join(f"- {n}" for n in notes),
             t_data(), t_files(), t_subsets(), t_main(), t_sweep(), t_fu2(), t_twelve(), t_oracle(), t_pilots(), t_rf()]
    return "\n\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if docs/RESULT_TABLES.md differs from a fresh build")
    a = ap.parse_args()
    text = build()
    if a.check:
        old = OUT.read_text() if OUT.exists() else ""
        if old != text:
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                                       "committed", "regenerated", n=1))
            print(f"FAIL: {OUT.relative_to(HERE)} differs from a fresh build")
            sys.exit(1)
        print(f"OK: {OUT.relative_to(HERE)} matches a fresh build from committed files; consistency checks passed")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(HERE)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
