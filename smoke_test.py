"""Fast synthetic end-to-end smoke test: Stages 0 -> 1 -> 1b -> 3 -> 4 on fake data.

Checks plumbing only (files exist, invariants hold, teacher unchanged). Numbers produced
here are NOT results. Takes about a minute on CPU.

  python smoke_test.py --work /tmp/nids_smoke
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="./smoke_work")
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
    print("\nSMOKE TEST PASSED (plumbing only - synthetic numbers are not results)")


if __name__ == "__main__":
    main()
