"""Fast synthetic end-to-end smoke test: Stages 0 -> 1 -> 1b -> 3 -> 4 on fake data.

Checks plumbing only (files exist, invariants hold, teacher unchanged). Numbers produced
here are NOT results. Takes about a minute on CPU.

  python smoke_test.py --work /tmp/nids_smoke
  python smoke_test.py --work W --write-golden g.json   # on the OLD code: record checkpoint hashes
  python smoke_test.py --work W --golden g.json         # on the NEW code: default flags must match
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def run(*args):
    cmd = [sys.executable, str(HERE / args[0]), *map(str, args[1:])]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ckpt_hashes(runs: Path, names=None) -> dict:
    """sha256 of tensors + classes + meta of every checkpoint (file bytes are not stable)."""
    import torch
    out = {}
    for p in sorted(runs.rglob("*.pt")):
        rel = str(p.relative_to(runs))
        if names is not None and rel not in names:
            continue
        ck = torch.load(p, map_location="cpu")
        h = hashlib.sha256()
        for k, v in sorted(ck["state"].items()):
            h.update(k.encode())
            h.update(v.numpy().tobytes())
        h.update(json.dumps([ck["classes"], ck["meta"]], sort_keys=True).encode())
        out[rel] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="./smoke_work")
    ap.add_argument("--golden", default=None, help="checkpoint hashes recorded with the previous code")
    ap.add_argument("--write-golden", default=None)
    a = ap.parse_args()
    w = Path(a.work).resolve()
    shutil.rmtree(w, ignore_errors=True)
    data, out = w / "input", w / "nids_out"
    out.mkdir(parents=True)

    run("make_synthetic.py", "--out", data, "--n-per-class", 1500, "--n-benign", 400)
    run("s0_audit.py", "--input", data, "--out", out)
    run("s1_preprocess.py", "--input", data, "--out", out, "--cap", 3000)
    sm = json.loads((out / "split_manifest.json").read_text())
    st = sm["stats"]
    # 300 copies of one Syn flow and 40 LDAP flows repeated in all 3 files must each count once
    assert st["raw_rows_by_label"]["Syn"] - st["nonfinite_rows_by_label"].get("Syn", 0) - st["distinct_rows_by_label"]["Syn"] >= 3 * 300 - 1
    assert st["raw_rows_by_label"]["LDAP"] - st["distinct_rows_by_label"]["LDAP"] >= 2 * 40
    assert st["kept_by_label"]["Syn"] <= 3000 and st["benign_distinct_kept"] > 0
    assert sm["cross_split_feature_overlap"] == 0 and st["n_conflict_vectors"] >= 1
    hs = [pd.read_parquet(out / f"clean_{sp}.parquet", columns=["_h"])["_h"] for sp in ("train", "calib", "val", "test")]
    assert pd.concat(hs).is_unique, "identical feature vector in two splits"
    (w / "schedule.json").write_text(json.dumps({"experiences": [["Syn"], ["LDAP"], ["UDP"]]}))
    s1b = ["s1b_experiences.py", "--out", out, "--schedule", w / "schedule.json",
           "--attack-cap", 1500, "--min-rows", 100, "--min-benign-protected", 50]
    # the exclusion guard must refuse rather than silently drop a lot of a class
    p = subprocess.run([sys.executable, str(HERE / s1b[0]), *map(str, s1b[1:]), "--scenario", "ci",
                        "--max-excluded-share", "0.0001"], capture_output=True, text=True)
    assert p.returncode != 0 and "refuse: exclusion would remove" in p.stderr, p.stderr[-500:]
    ems = {}
    for sc in ("ci", "cii"):
        run(*s1b, "--scenario", sc)
        ems[sc] = em = json.loads((out / sc / "experience_manifest.json").read_text())
        assert "UDP" not in em["counts_per_experience"]["E0"] and "UDP" not in em["counts_per_experience"]["E1"]
        assert all(v["equal_to_a_train_input"] == 0 for v in em["scaled_input_overlap"].values())
    pe = json.loads((out / "protected_exclusions.json").read_text())
    assert ems["ci"]["protected_exclusions"]["rids_sha256"] == ems["cii"]["protected_exclusions"]["rids_sha256"] \
        == pe["rids_sha256"], "CI and CII would score different protected rows"
    causes = {s for x in pe["diagnosis"].values() for d in x.values() for s in d["first_step"]}
    # synthetic twins: -0.0 (signed_zero), clip, float32, unseen protocol (model_input)
    assert {"signed_zero", "clip", "float32", "model_input"} <= causes, causes
    assert pe["n_excluded"] > 0 and all(v["equal_to_a_train_input"] == 0
                                        for x in pe["residual"].values() for v in x.values())
    fast = ["--epochs0", 3, "--epochs-update", 2, "--d", 16, "--layers", 1,
            "--mem-per-class", 200, "--device", "cpu", "--poison-frac", 0.5]
    for sc in ("ci", "cii"):  # dry run: verifies inputs against Stage 1b, writes nothing
        run("s3_continual.py", "--out", out, "--scenario", sc, "--run", "dry", *fast, "--dry-run")
        assert not (out / "runs" / sc / "dry").exists()
    # strict (default) gate: exercises reject -> R1 -> keep previous model
    run("s3_continual.py", "--out", out, "--scenario", "cii", "--run", "smoke", *fast)
    run("s4_report.py", "--out", out, "--scenario", "cii", "--run", "smoke")
    # open gate: exercises accept -> buffer grows; CI scenario too
    run("s3_continual.py", "--out", out, "--scenario", "ci", "--run", "smoke_open", *fast,
        "--gate-old-drop", 1, "--gate-new-min", 0, "--gate-far-max", 1, "--gate-newfar-max", 1)
    run("s4_report.py", "--out", out, "--scenario", "ci", "--run", "smoke_open")
    ro = out / "runs" / "ci" / "smoke_open"
    for arm in ("clean", "poisoned", "poisoned_repair"):
        recs = json.loads((ro / arm / "rounds.json").read_text())
        assert all(x["accepted"] and x["attempt"] == "C1" for x in recs)
    b1 = pd.read_csv(ro / "clean" / "buffer_before_E1.csv")
    b2 = pd.read_csv(ro / "clean" / "buffer_before_E2.csv")
    assert set(b2["label"]) > set(b1["label"]), "buffer did not grow after acceptance"

    r = out / "runs" / "cii" / "smoke"
    need = ["T.pt", "A0.pt", "A0_metrics.json", "config.json", "run_manifest.json",
            "report/attempts.csv", "report/per_class_val.csv", "report/final_test.csv",
            "report/target_recall.png", "report/per_class_E1.png", "report/per_class_E2.png"]
    for arm in ("clean", "poisoned", "poisoned_repair"):
        need += [f"{arm}/rounds.json", f"{arm}/final_test.json", f"{arm}/A1.pt", f"{arm}/A2.pt",
                 f"{arm}/E1_C1.pt", f"{arm}/buffer_before_E1.csv"]
    missing = [p for p in need if not (r / p).exists()]
    assert not missing, f"missing outputs: {missing}"
    assert json.loads((r / "run_manifest.json").read_text())["T_unchanged"]
    cfg = json.loads((r / "config.json").read_text())  # Stage 3 scores only the rows Stage 1b kept
    for sp in ("val", "test"):
        kept = {k: v["kept"] for k, v in pe["by_split_class"][sp].items()}
        assert cfg[f"{sp}_rows"] == kept, (sp, cfg[f"{sp}_rows"], kept)
    clean_buf = pd.read_csv(r / "clean" / "buffer_before_E1.csv")
    pois_buf = pd.read_csv(r / "poisoned" / "buffer_before_E1.csv")
    assert clean_buf["flipped"].sum() == 0 and pois_buf["flipped"].sum() > 0
    assert (pois_buf.loc[pois_buf["flipped"], "true_label"].nunique() == 1), "poison touched >1 class"
    for arm in ("clean", "poisoned", "poisoned_repair"):
        for rec in json.loads((r / arm / "rounds.json").read_text()):
            if rec["attempt"] == "R1":
                assert arm == "poisoned_repair"
                prev = [x for x in json.loads((r / arm / "rounds.json").read_text())
                        if x["experience"] == rec["experience"] and x["attempt"] == "C1"]
                assert prev and not prev[0]["accepted"], "R1 ran although C1 was accepted"
    # every arm is tested on the same protected rows (all scheduled classes, learned or not)
    fin = pd.read_csv(r / "report" / "final_test.csv")
    assert fin["test_rows_sha256"].nunique() == 1 and "recall_UDP" in fin.columns
    # --resume with a changed setting must refuse instead of mixing runs
    p = subprocess.run([sys.executable, str(HERE / "s3_continual.py"), "--out", str(out), "--scenario",
                        "cii", "--run", "smoke", *map(str, fast), "--resume", "--poison-frac", "0.25"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "refuse --resume" in p.stderr, p.stderr[-500:]

    # default flags must reproduce the previous code's checkpoints exactly (main stays reproducible)
    runs = out / "runs"
    base = ckpt_hashes(runs)
    if a.write_golden:
        Path(a.write_golden).write_text(json.dumps(base, indent=0))
    if a.golden:
        gold = json.loads(Path(a.golden).read_text())
        gold = {k: v for k, v in gold.items() if k.endswith(".pt")}
        diff = [k for k in gold if base.get(k) != gold[k]]
        assert not diff and len(gold) == len(base), f"default-flag checkpoints changed: {diff}"
        print(f"default flags: {len(gold)} checkpoint hashes identical to the previous code")
    # a completed run is never overwritten
    p = subprocess.run([sys.executable, str(HERE / "s3_continual.py"), "--out", str(out), "--scenario",
                        "cii", "--run", "smoke", *map(str, fast)], capture_output=True, text=True)
    assert p.returncode != 0 and "never overwritten" in p.stderr, p.stderr[-500:]

    # new flags: seed, conf gate, three-way repair + relabel; then the same + forced R1 diagnostics
    new = [*fast, "--seed", 43, "--gate-conf-drop", 0.05, "--repair-rule", "threeway", "--relabel-to-teacher"]
    run("s3_continual.py", "--out", out, "--scenario", "cii", "--run", "flags", *new)
    run("s3_continual.py", "--out", out, "--scenario", "cii", "--run", "flags_diag", *new,
        "--forced-r1", "soft,threeway,threeway+relabel")
    f1, f2 = runs / "cii" / "flags", runs / "cii" / "flags_diag"
    h1 = ckpt_hashes(f1)
    assert h1 == ckpt_hashes(f2, set(h1)), "forced R1 diagnostics changed a decision path"
    assert ckpt_hashes(f1, {"T.pt"}) != ckpt_hashes(runs / "cii" / "smoke", {"T.pt"}), "--seed had no effect"
    assert json.loads((f1 / "config.json").read_text())["seed"] == 43
    for arm in ("clean", "poisoned"):
        d = json.loads((f2 / arm / "r1_diag.json").read_text())
        assert d and all(x["diagnostic"] and x["attempt"] == "R1_diag" for x in d)
        assert {x["method"] for x in d} == {"soft", "threeway", "threeway+relabel"}
    assert not (f2 / "poisoned_repair" / "r1_diag.json").exists()
    recs = json.loads((f1 / "poisoned_repair" / "rounds.json").read_text())
    assert all("conf_check" in r and "mean_conf" in r["val"] for r in recs)
    r1 = [r for r in recs if r["attempt"] == "R1"]
    assert all(r["teacher_weights"]["method"] == "threeway+relabel" for r in r1)
    print(f"new flags ran: R1 attempts {len(r1)}, conf-gate rejections "
          f"{sum(any('mean p' in g for g in r['gate_reasons']) for r in recs)}")
    run("s4_report.py", "--out", out, "--scenario", "cii", "--run", "flags_diag")

    # a second schedule under --tag never touches the first schedule's Stage 1b files
    before = {p: p.read_bytes() for p in [out / "protected_exclusions.json", out / "cii" / "scaler.json"]}
    (w / "schedule_alt.json").write_text(json.dumps({"experiences": [["Syn"], ["LDAP", "UDP"]]}))
    run("s1b_experiences.py", "--out", out, "--schedule", w / "schedule_alt.json", "--scenario", "cii",
        "--tag", "alt", "--attack-cap", 1500, "--min-rows", 100, "--min-benign-protected", 50)
    assert all(p.read_bytes() == b for p, b in before.items()), "--tag overwrote the original Stage 1b files"
    run("s3_continual.py", "--out", out, "--scenario", "cii", "--tag", "alt", "--run", "alt", *fast,
        "--gate-old-drop", 1, "--gate-new-min", 0, "--gate-far-max", 1, "--gate-newfar-max", 1)
    assert "alt/cii/scaler.json" in json.loads((runs / "cii" / "alt" / "config.json").read_text())["inputs"]

    # follow-up FU1 (s9): reads a finished run, never writes into it; base recipe = the logged s3 C1
    src_run = runs / "cii" / "smoke"
    before = {p: p.read_bytes() for p in src_run.rglob("*") if p.is_file()}
    fu = ["--src", out, "--work", w / "fu", "--scenario", "cii", "--tag", "", "--source-run", "smoke",
          "--device", "cpu", "--seeds", "42,43"]
    run("s9_followup.py", "separability", *fu, "--pairs", "LDAP:UDP")
    run("s9_followup.py", "probe", *fu, "--block", 1, "--recipes", "base,ep10,lr3e-4,balanced")
    run("s9_followup.py", "probe", *fu, "--block", 2, "--recipes", "base,balanced,distill")
    fdir = w / "fu" / "followup" / "cii"
    rep = json.loads((fdir / "probe_E1" / "base_s42.json").read_text())["reproduces_logged_C1"]
    assert rep == 0.0, f"s9 base recipe does not reproduce s3's C1 (max diff {rep})"
    run("s9_followup.py", "decide", "--work", w / "fu", "--new-class-block", 1, "--old-class-block", 2)
    forced = w / "fu" / "forced_decisions.json"  # forced recipes, so C2 and D1 are exercised
    forced.write_text(json.dumps({"new_class": {"recipe": "ep10"}, "old_class": {"recipe": "distill"}}))
    run("s9_followup.py", "policy", *fu, "--decisions", forced, "--run", "fu_smoke")
    recs = json.loads((fdir / "fu_smoke" / "rounds.json").read_text())
    logged = json.loads((src_run / "clean" / "rounds.json").read_text())[0]
    assert recs[0]["attempt"] == "C1" and recs[0]["val"]["recall"] == logged["val"]["recall"], "policy C1 != s3 C1"
    for b in {r["block"] for r in recs}:
        att = [r for r in recs if r["block"] == b]
        assert len(att) <= 3 and sum(r["accepted"] for r in att) <= 1 and not any(r["accepted"] for r in att[:-1])
        assert all(r["attempt"] in ("C2", "D1") for r in att[1:]), att
    p = subprocess.run([sys.executable, str(HERE / "s9_followup.py"), "score-test", *map(str, fu), "--run", "fu_smoke"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "decisions-commit" in p.stderr, p.stderr[-500:]
    run("s9_followup.py", "score-test", *fu, "--run", "fu_smoke", "--decisions-commit", "0000000")
    p = subprocess.run([sys.executable, str(HERE / "s9_followup.py"), "score-test", *map(str, fu), "--run", "fu_smoke",
                        "--decisions-commit", "0000000"], capture_output=True, text=True)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    assert before == {p: p.read_bytes() for p in src_run.rglob("*") if p.is_file()}, "s9 changed the source run"
    print(f"FU1 (s9) ran: {len(recs)} policy attempts "
          f"({', '.join(sorted({r['attempt'] for r in recs}))}); source run unchanged")
    print("\nSMOKE TEST PASSED (plumbing only - synthetic numbers are not results)")


if __name__ == "__main__":
    main()
