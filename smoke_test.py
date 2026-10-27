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
import zipfile
from pathlib import Path

import numpy as np
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
    sep = pd.read_csv(w / "fu" / "followup" / "cii" / "separability_exact.csv")
    assert (sep["distance"] == "float64_exact").all() and not (w / "fu" / "followup" / "cii" / "separability.csv").exists()
    run("s9_followup.py", "probe", *fu, "--block", 1, "--recipes", "base,ep10,lr3e-4,balanced")
    run("s9_followup.py", "probe", *fu, "--block", 2, "--recipes", "base,balanced,distill")
    fdir = w / "fu" / "followup" / "cii"
    rep = json.loads((fdir / "probe_E1" / "base_s42.json").read_text())["reproduces_logged_C1"]
    assert rep == 0.0, f"s9 base recipe does not reproduce s3's C1 (max diff {rep})"
    run("s9_followup.py", "decide", "--work", w / "fu", "--new-class-block", 1, "--old-class-block", 2)
    forced = fdir.parent / "decisions.json"  # overwritten with forced recipes, so C2 and D1 are exercised
    forced.write_text(json.dumps({"new_class": {"recipe": "ep10"}, "old_class": {"recipe": "distill"}}))
    run("s9_followup.py", "policy", *fu, "--decisions", forced, "--run", "fu_smoke")
    recs = json.loads((fdir / "fu_smoke" / "rounds.json").read_text())
    logged = json.loads((src_run / "clean" / "rounds.json").read_text())[0]
    assert recs[0]["attempt"] == "C1" and recs[0]["val"]["recall"] == logged["val"]["recall"], "policy C1 != s3 C1"
    for b in {r["block"] for r in recs}:
        att = [r for r in recs if r["block"] == b]
        assert len(att) <= 3 and sum(r["accepted"] for r in att) <= 1 and not any(r["accepted"] for r in att[:-1])
        assert all(r["attempt"] in ("C2", "D1") for r in att[1:]), att
    def s9(*args):
        return subprocess.run([sys.executable, str(HERE / "s9_followup.py"), *map(str, args)],
                              capture_output=True, text=True)
    p = s9("score-test", *fu, "--run", "fu_smoke")
    assert p.returncode != 0 and "decisions-commit" in p.stderr, p.stderr[-500:]
    # validation -> test handoff: bundle (no test inside), commit the validation files, restore elsewhere, score
    bz = w / "fu1_restore.zip"
    run("s9_followup.py", "bundle", "--work", w / "fu", "--bundle", bz)
    names = zipfile.ZipFile(bz).namelist()
    assert "RESTORE_MANIFEST.json" in names and not any("final_test" in n for n in names), names
    repo = w / "repo"
    dst = repo / "experiments" / "results" / "05_fu1" / "followup"
    for f in ["decisions.json", "cii/fu_smoke/summary.json", "cii/fu_smoke/rounds.json"]:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fdir.parent / f, dst / f)
    git = ["git", "-C", str(repo), "-c", "user.name=smoke", "-c", "user.email=smoke@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fu1 val"], check=True)
    sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    bad = w / "bad_bundle"  # a damaged bundle must be refused
    zipfile.ZipFile(bz).extractall(bad)
    with open(bad / "followup" / "cii" / "fu_smoke" / "final.pt", "ab") as f:
        f.write(b"x")
    p = s9("restore", "--work", w / "fu_bad", "--bundle", bad)
    assert p.returncode != 0 and "bundle damaged" in p.stderr, p.stderr[-500:]
    fu2 = [*fu[:2], "--work", w / "fu2", *fu[4:]]  # a fresh session: nothing but the bundle
    run("s9_followup.py", "restore", "--work", w / "fu2", "--bundle", bz)
    p = s9("score-test", *fu2, "--run", "fu_smoke", "--decisions-commit", "0" * 7, "--repo", repo)
    assert p.returncode != 0 and "not found in commit" in p.stderr, p.stderr[-500:]
    run("s9_followup.py", "score-test", *fu2, "--run", "fu_smoke", "--decisions-commit", sha, "--repo", repo)
    p = s9("score-test", *fu2, "--run", "fu_smoke", "--decisions-commit", sha, "--repo", repo)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    assert before == {p: p.read_bytes() for p in src_run.rglob("*") if p.is_file()}, "s9 changed the source run"
    print(f"FU1 (s9) ran: {len(recs)} policy attempts "
          f"({', '.join(sorted({r['attempt'] for r in recs}))}); source run unchanged")

    # follow-up FU2 (s10): relabel R1 at f = 1.0 from a sweep-like source run; E1 replays it exactly
    run("s3_continual.py", "--out", out, "--scenario", "cii", "--run", "sweep_f1.0_s42", *fast,
        "--poison-frac", 1.0, "--forced-r1", "soft,threeway,threeway+relabel")
    src_run = runs / "cii" / "sweep_f1.0_s42"
    before = {p: p.read_bytes() for p in src_run.rglob("*") if p.is_file()}
    f2 = ["--src", out, "--seeds", 42, "--device", "cpu"]
    run("s10_relabel.py", "run", *f2, "--work", w / "r2", "--dry-run")
    assert not (w / "r2").exists(), "dry run wrote files"
    run("s10_relabel.py", "run", *f2, "--work", w / "r2")
    rd = w / "r2" / "fu2" / "cii" / "fu2_relabel_f1.0_s42"
    summ = json.loads((rd / "summary.json").read_text())
    ck = summ["p0_checks"]
    assert ck["buffer_E1_equals_source"] and ck["E1_C1_equals_source_C1"] and ck["E1_R1_equals_forced_diagnostic"] in (True, None), ck
    recs = json.loads((rd / "poisoned_relabel" / "rounds.json").read_text())
    assert all(r["teacher_weights"]["method"] == "threeway+relabel" for r in recs if r["attempt"] == "R1")
    assert not list(rd.rglob("final_test.json")), "s10 run scored test"
    run("s10_relabel.py", "run", *f2, "--work", w / "r2")  # finished seed: skipped, never overwritten
    assert json.loads((rd / "summary.json").read_text()) == summ
    def s10(*args):
        return subprocess.run([sys.executable, str(HERE / "s10_relabel.py"), *map(str, args)],
                              capture_output=True, text=True)
    bz = w / "fu2_restore.zip"
    run("s10_relabel.py", "bundle", "--work", w / "r2", "--bundle", bz)
    assert not any("final_test" in n for n in zipfile.ZipFile(bz).namelist())
    dst = repo / "experiments" / "results" / "06_fu2"
    for f in ["summary.json", "poisoned_relabel/rounds.json"]:
        (dst / "fu2" / "cii" / rd.name / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rd / f, dst / "fu2" / "cii" / rd.name / f)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fu2 val"], check=True)
    sha2 = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    run("s10_relabel.py", "restore", "--work", w / "r2b", "--bundle", bz)
    p = s10("score-test", *f2, "--work", w / "r2b", "--decisions-commit", sha, "--repo", repo)
    assert p.returncode != 0 and "not found in commit" in p.stderr, p.stderr[-500:]
    run("s10_relabel.py", "score-test", *f2, "--work", w / "r2b", "--decisions-commit", sha2, "--repo", repo)
    p = s10("score-test", *f2, "--work", w / "r2b", "--decisions-commit", sha2, "--repo", repo)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    assert before == {p: p.read_bytes() for p in src_run.rglob("*") if p.is_file()}, "s10 changed the source run"
    print(f"FU2 (s10) ran: {summ['update_status']}, P0 {ck}; source run unchanged")

    # joint-training oracle (s11) on an all12-like tag: same val/test rows as the continual runs, test once
    (w / "schedule_all12.json").write_text(json.dumps({"experiences": [["Syn"], ["LDAP"], ["UDP"]]}))
    for sc in ("cii", "ci"):
        run("s1b_experiences.py", "--out", out, "--schedule", w / "schedule_all12.json", "--scenario", sc,
            "--tag", "all12", "--attack-cap", 1500, "--min-rows", 100, "--min-benign-protected", 50)
        run("s3_continual.py", "--out", out, "--scenario", sc, "--tag", "all12", "--run", f"all12_{sc}_s42", *fast)
    before = {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    o = ["--src", out, "--device", "cpu"]
    run("s11_oracle.py", "train", *o, "--work", w / "or")
    run("s11_oracle.py", "train", *o, "--work", w / "or")  # resumable: every cell skipped
    run("s11_oracle.py", "decide", "--work", w / "or")
    od = w / "or" / "oracle"
    v = json.loads((od / "cii" / "e10_s42" / "val.json").read_text())
    assert v["train_rows"] == len(np.unique(np.concatenate(list(np.load(out / "all12" / "cii" / "experience_rids.npz").values()))))
    assert not list(od.rglob("test.json")), "s11 train scored test"
    def s11(*args):
        return subprocess.run([sys.executable, str(HERE / "s11_oracle.py"), *map(str, args)],
                              capture_output=True, text=True)
    bz = w / "oracle_restore.zip"
    run("s11_oracle.py", "bundle", "--work", w / "or", "--bundle", bz)
    assert not any("test" in n for n in zipfile.ZipFile(bz).namelist())
    dst = repo / "experiments" / "results" / "07_oracle"
    for p in od.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".csv"):
            (dst / "oracle" / p.relative_to(od)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst / "oracle" / p.relative_to(od))
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "oracle val"], check=True)
    sha3 = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    run("s11_oracle.py", "restore", "--work", w / "or2", "--bundle", bz)
    p = s11("score-test", *o, "--work", w / "or2", "--decisions-commit", sha2, "--repo", repo)
    assert p.returncode != 0 and "not found in commit" in p.stderr, p.stderr[-500:]
    run("s11_oracle.py", "score-test", *o, "--work", w / "or2", "--decisions-commit", sha3, "--repo", repo)
    p = s11("score-test", *o, "--work", w / "or2", "--decisions-commit", sha3, "--repo", repo)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    assert before == {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}, "s11 changed a run"
    print(f"oracle (s11) ran: {len(list(od.rglob('val.json')))} cells; runs unchanged")

    # repeated corruption (s12): more flips before every update, flips persist, validation only, test once
    rc = ["--src", out, "--classes", "LDAP,NTP,SNMP", "--attack-cap", 1500, "--device", "cpu",
          "--s1b-args", "--min-rows 100 --min-benign-protected 50"]
    small = "--epochs0 3 --epochs-update 2 --d 16 --layers 1 --mem-per-class 200"
    open_gate = " --gate-old-drop 1 --gate-new-min 0 --gate-far-max 1 --gate-newfar-max 1"
    def s12(*args):
        return subprocess.run([sys.executable, str(HERE / "s12_repeated.py"), *map(str, args)],
                              capture_output=True, text=True)
    rs = w / "rc_strict"  # registered gate: the pilot decides (proceed or stop); run obeys it
    run("s12_repeated.py", "prepare", *rc, "--work", rs)
    run("s12_repeated.py", "pilot", *rc, "--work", rs, "--s3-args", small)
    pj = json.loads((rs / "rc" / "pilot.json").read_text())
    if not pj["proceed"]:
        p = s12("run", *rc, "--work", rs, "--s3-args", small)
        assert p.returncode != 0 and "pilot said stop" in p.stderr, p.stderr[-500:]
    ro = w / "rc_open"  # open gate: every update accepted, so the full flow runs
    run("s12_repeated.py", "prepare", *rc, "--work", ro)
    run("s12_repeated.py", "pilot", *rc, "--work", ro, "--s3-args", small + open_gate)
    run("s12_repeated.py", "run", *rc, "--work", ro, "--s3-args", small + open_gate)
    run("s12_repeated.py", "run", *rc, "--work", ro, "--s3-args", small + open_gate)  # resumable: all skipped
    run("s12_repeated.py", "summarize", *rc, "--work", ro)
    rruns = sorted((ro / "runs" / "cii").iterdir())
    assert len(rruns) == 6 and not list(ro.rglob("final_test.json")), "s12 run scored test"
    for r in rruns:
        assert "A0_test" not in json.loads((r / "run_manifest.json").read_text())
        assert (r / "poisoned" / "r1_diag.json").exists() and not (r / "clean" / "r1_diag.json").exists()
        rho = float(r.name.split("_r")[-1].split("_")[0])
        for t in (1, 2, 3):  # 20 or 10 more of the 200 stored Syn rows per update, never unflipped
            b = pd.read_csv(r / "poisoned_repair" / f"buffer_before_E{t}.csv")
            assert b["flipped"].sum() == t * round(rho * 200) and (b.loc[b["flipped"], "true_label"] == "Syn").all()
    summ = json.loads((ro / "rc" / "summary.json").read_text())
    assert all(summ["clean_arm_identical_across_rho"].values()), "clean arm differs between rho runs"
    assert summ["predictions_val"]["rho_0.10"]["RC3_repair_is_gated"]["met"], "repair deployed other models"
    assert len(pd.read_csv(ro / "rc" / "trajectory_val.csv")) == 2 * 3 * 3 * 4
    vz, bz = w / "rc_val.zip", w / "rc_restore.zip"
    run("s12_repeated.py", "zip-val", *rc, "--work", ro, "--bundle", vz)
    run("s12_repeated.py", "bundle", *rc, "--work", ro, "--bundle", bz)
    assert not any("test" in n for n in zipfile.ZipFile(bz).namelist())
    zipfile.ZipFile(vz).extractall(repo / "experiments" / "results" / "08_rc")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "rc val"], check=True)
    sha4 = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    ro2 = w / "rc_restored"  # a fresh session: restore first, then prepare rebuilds Stage 1b from the pilot's tags
    run("s12_repeated.py", "restore", *rc, "--work", ro2, "--bundle", bz)
    run("s12_repeated.py", "prepare", *rc, "--work", ro2)
    p = s12("score-test", *rc, "--work", ro2, "--decisions-commit", sha3, "--repo", repo)
    assert p.returncode != 0 and "not found in commit" in p.stderr, p.stderr[-500:]
    run("s12_repeated.py", "score-test", *rc, "--work", ro2, "--decisions-commit", sha4, "--repo", repo)
    p = s12("score-test", *rc, "--work", ro2, "--decisions-commit", sha4, "--repo", repo)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    tt = pd.read_csv(ro2 / "rc" / "test" / "test_trajectory.csv")
    assert len(tt) == 2 * 3 * 3 * 4 and tt["test_rows_sha256"].nunique() == 1
    print(f"repeated corruption (s12) ran: pilot proceed={pj['proceed']} (registered gate), "
          f"{len(rruns)} runs, test scored once after the commit")

    # RCS (s13): clean screen + confirmation decide before any poisoned run; s12's code with the rcs/ layout
    rcs = ["--src", out, "--candidates", "c3=LDAP,NTP,SNMP;c2=NTP,SNMP", "--attack-cap", 1500, "--device", "cpu",
           "--s1b-args", "--min-rows 100 --min-benign-protected 50"]
    loose = ["--min-margin", "-9", "--syn-band", "9"]
    def s13(*args):
        return subprocess.run([sys.executable, str(HERE / "s13_rcs.py"), *map(str, args)],
                              capture_output=True, text=True)
    ss = w / "rcs_strict"  # registered gate and predicate: stages S and C decide; confirm / run obey them
    run("s13_rcs.py", "prepare", *rcs, "--work", ss)
    run("s13_rcs.py", "screen", *rcs, "--work", ss, "--s3-args", small)
    sj = json.loads((ss / "rcs" / "screen.json").read_text())
    if not sj["selected"]:
        p = s13("confirm", *rcs, "--work", ss, "--s3-args", small)
        assert p.returncode != 0 and "selected no candidate" in p.stderr, p.stderr[-500:]
    else:
        run("s13_rcs.py", "confirm", *rcs, "--work", ss, "--s3-args", small)
    gj = ss / "rcs" / "go.json"
    if not (gj.exists() and json.loads(gj.read_text())["proceed"]):
        p = s13("run", *rcs, "--work", ss, "--s3-args", small)
        assert p.returncode != 0 and "no go.json with proceed" in p.stderr, p.stderr[-500:]
        assert not list((ss / "runs" / "cii").glob("*/poisoned")), "a poisoned arm ran without go"
    so = w / "rcs_open"  # open gate and loose predicate: the full flow
    run("s13_rcs.py", "prepare", *rcs, "--work", so)
    p = s13("run", *rcs, "--work", so, "--s3-args", small + open_gate)
    assert p.returncode != 0 and "no go.json with proceed" in p.stderr, p.stderr[-500:]
    run("s13_rcs.py", "screen", *rcs, *loose, "--work", so, "--s3-args", small + open_gate)
    assert json.loads((so / "rcs" / "screen.json").read_text())["selected"] == "c3", "priority order not used"
    assert not list((so / "runs" / "cii").glob("*/poisoned")), "the screen trained a poisoned arm"
    run("s13_rcs.py", "confirm", *rcs, *loose, "--work", so, "--s3-args", small + open_gate)
    g = json.loads((so / "rcs" / "go.json").read_text())
    assert g["proceed"] and g["tag"] == "c3" and not list((so / "runs" / "cii").glob("*/poisoned"))
    run("s13_rcs.py", "run", *rcs, "--work", so, "--s3-args", small + open_gate)
    run("s13_rcs.py", "run", *rcs, "--work", so, "--s3-args", small + open_gate)  # resumable: all skipped
    run("s13_rcs.py", "summarize", *rcs, "--work", so)  # asserts the confirmation clean files are unchanged
    names = sorted(r.name for r in (so / "runs" / "cii").iterdir())
    assert names == sorted([f"rcs_{t}_r0.10_s{s}" for t in ("c3", "c2") for s in (42, 43, 44)] +
                           [f"rcs_c3_r{r}_s{s}" for r in ("0.10", "0.05") for s in (45, 46, 47)]), names
    assert not (so / "rc").exists() and not list(so.rglob("final_test.json")), "wrong layout or test scored"
    summ = json.loads((so / "rcs" / "summary.json").read_text())
    assert sorted(summ["runs"]) == sorted(f"rcs_c3_r{r}_s{s}" for r in ("0.10", "0.05") for s in (45, 46, 47))
    assert all(summ["clean_arm_identical_across_rho"].values()), "clean arm differs between rho runs"
    assert summ["predictions_val"]["rho_0.10"]["RC3_repair_is_gated"]["met"]
    vz, bz = w / "rcs_val.zip", w / "rcs_restore.zip"
    run("s13_rcs.py", "zip-val", *rcs, "--work", so, "--bundle", vz)
    run("s13_rcs.py", "bundle", *rcs, "--work", so, "--bundle", bz)
    vn = zipfile.ZipFile(vz).namelist()
    assert "rcs/screen.json" in vn and "rcs/go.json" in vn and not any(n.endswith(".pt") or "test" in n for n in vn)
    assert any("_s42/clean/rounds.json" in n for n in vn), "screening runs missing from rcs_val.zip"
    zipfile.ZipFile(vz).extractall(repo / "experiments" / "results" / "09_rcs")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "rcs val"], check=True)
    sha5 = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    so2 = w / "rcs_restored"  # a fresh session: restore first, then prepare rebuilds both candidate tags
    run("s13_rcs.py", "restore", *rcs, "--work", so2, "--bundle", bz)
    run("s13_rcs.py", "prepare", *rcs, "--work", so2)
    p = s13("score-test", *rcs, "--work", so2, "--decisions-commit", sha4, "--repo", repo)
    assert p.returncode != 0 and "not found in commit" in p.stderr, p.stderr[-500:]
    run("s13_rcs.py", "score-test", *rcs, "--work", so2, "--decisions-commit", sha5, "--repo", repo)
    p = s13("score-test", *rcs, "--work", so2, "--decisions-commit", sha5, "--repo", repo)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    tt = pd.read_csv(so2 / "rcs" / "test" / "test_trajectory.csv")
    assert len(tt) == 2 * 3 * 3 * 4 and tt["test_rows_sha256"].nunique() == 1 and set(tt["seed"]) == {45, 46, 47}
    print(f"RCS (s13) ran: strict screen selected={sj['selected']}; open flow screen -> confirm -> 6 runs, "
          f"test scored once after the commit")

    # RF (s14): fixed-class refresh (a class repeats with disjoint rows; no new class after E0); strict clean pilot
    rf = ["--src", out, "--refresh-class", "NTP", "--updates", 3, "--attack-cap", 1500, "--device", "cpu",
          "--s1b-args", "--min-rows 100 --min-benign-protected 50"]
    def s14(*args):
        return subprocess.run([sys.executable, str(HERE / "s14_refresh.py"), *map(str, args)],
                              capture_output=True, text=True)
    fs = w / "rf_strict"  # registered gate and predicate: the pilot decides; a failed pilot is saved and zipped
    run("s14_refresh.py", "prepare", *rf, "--work", fs)
    em = json.loads((fs / "ref6" / "cii" / "experience_manifest.json").read_text())
    assert em["schedule"] == [["Syn", "NTP"], ["NTP"], ["NTP"], ["NTP"]]
    z = np.load(fs / "ref6" / "cii" / "experience_rids.npz")
    ids = np.concatenate([z[k] for k in z.files])
    assert len(ids) == len(np.unique(ids)) and all(em["counts_per_experience"][f"E{t}"]["NTP"] > 0 for t in range(4))
    run("s14_refresh.py", "pilot", *rf, "--work", fs, "--s3-args", small)
    fp = json.loads((fs / "rf" / "pilot.json").read_text())
    for r in json.loads((fs / "runs" / "cii" / "rf_ref6_r0.10_s42" / "clean" / "rounds.json").read_text()):
        assert r["new_classes"] == [], "a refresh update had a new class"
    if not fp["proceed"]:
        p = s14("run", *rf, "--work", fs, "--s3-args", small)
        assert p.returncode != 0 and "no pilot.json with proceed" in p.stderr, p.stderr[-500:]
        assert not list((fs / "runs" / "cii").glob("*/poisoned")), "a poisoned arm ran after a failed pilot"
        run("s14_refresh.py", "zip-val", *rf, "--work", fs, "--bundle", w / "rf_clean.zip")
        fz = zipfile.ZipFile(w / "rf_clean.zip").namelist()
        assert "rf/pilot.json" in fz and sum(n.endswith("clean/rounds.json") for n in fz) == 3, fz
    fo = w / "rf_open"  # open gate and loose predicate: the full flow
    run("s14_refresh.py", "prepare", *rf, "--work", fo)
    run("s14_refresh.py", "pilot", *rf, *loose, "--work", fo, "--s3-args", small + open_gate)
    assert json.loads((fo / "rf" / "pilot.json").read_text())["proceed"]
    assert not list((fo / "runs" / "cii").glob("*/poisoned")), "the pilot trained a poisoned arm"
    run("s14_refresh.py", "run", *rf, "--work", fo, "--s3-args", small + open_gate)
    run("s14_refresh.py", "run", *rf, "--work", fo, "--s3-args", small + open_gate)  # resumable: all skipped
    run("s14_refresh.py", "summarize", *rf, "--work", fo)  # asserts the pilot's clean files are unchanged
    names = sorted(r.name for r in (fo / "runs" / "cii").iterdir())
    assert names == sorted(f"rf_ref6_r{r}_s{s}" for r in ("0.10", "0.05") for s in (42, 43, 44)), names
    assert not (fo / "rc").exists() and not (fo / "rcs").exists() and not list(fo.rglob("final_test.json"))
    for r in names:
        b = pd.read_csv(fo / "runs" / "cii" / r / "poisoned" / "buffer_before_E3.csv")
        rho = float(r.split("_r")[-1].split("_")[0])
        assert b["flipped"].sum() == 3 * round(rho * 200) and (b.loc[b["flipped"], "true_label"] == "Syn").all()
    summ = json.loads((fo / "rf" / "summary.json").read_text())
    assert all(summ["clean_arm_identical_across_rho"].values()) and summ["predictions_val"]["rho_0.10"]["RC3_repair_is_gated"]["met"]
    tv = pd.read_csv(fo / "rf" / "trajectory_val.csv")
    assert len(tv) == 2 * 3 * 3 * 4 and set(tv.loc[tv["t"] > 0, "new_class"]) == {"refresh"}
    vz, bz = w / "rf_val.zip", w / "rf_restore.zip"
    run("s14_refresh.py", "zip-val", *rf, "--work", fo, "--bundle", vz)
    run("s14_refresh.py", "bundle", *rf, "--work", fo, "--bundle", bz)
    vn = zipfile.ZipFile(vz).namelist()
    assert "rf/pilot.json" in vn and not any(n.endswith(".pt") or "test" in n for n in vn)
    zipfile.ZipFile(vz).extractall(repo / "experiments" / "results" / "10_rf")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "rf val"], check=True)
    sha6 = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    fo2 = w / "rf_restored"  # a fresh session: restore first, then prepare rebuilds Stage 1b
    run("s14_refresh.py", "restore", *rf, "--work", fo2, "--bundle", bz)
    run("s14_refresh.py", "prepare", *rf, "--work", fo2)
    p = s14("score-test", *rf, "--work", fo2, "--decisions-commit", sha5, "--repo", repo)
    assert p.returncode != 0 and "not found in commit" in p.stderr, p.stderr[-500:]
    run("s14_refresh.py", "score-test", *rf, "--work", fo2, "--decisions-commit", sha6, "--repo", repo)
    p = s14("score-test", *rf, "--work", fo2, "--decisions-commit", sha6, "--repo", repo)
    assert p.returncode != 0 and "scored once" in p.stderr, p.stderr[-500:]
    tt = pd.read_csv(fo2 / "rf" / "test" / "test_trajectory.csv")
    assert len(tt) == 2 * 3 * 3 * 4 and tt["test_rows_sha256"].nunique() == 1
    print(f"RF (s14) ran: strict pilot proceed={fp['proceed']}; open flow pilot -> 6 runs, test once after the commit")
    print("\nSMOKE TEST PASSED (plumbing only - synthetic numbers are not results)")


if __name__ == "__main__":
    main()
