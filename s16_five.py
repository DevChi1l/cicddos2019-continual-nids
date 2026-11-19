"""FIVE (POST HOC, 2026-11-19): does the four-class defence carry over to a fifth class?
Pre-registration: experiments/prereg_five.md.

Schedule (tag `five`, CII): E0 = BENIGN + Syn, E1 + LDAP, E2 + UDP, E3 + TFTP. Same recipe, gate, buffer and attack
as the four-class sweep / FU2 (s3 defaults; attack cap 13,890 = the full cleaned LDAP training count, as before).

  prepare   copy the Stage 1 files from notebook 04's output and build Stage 1b for tag five (s1b, CII)
  pilot     clean arm only, seeds 42-44 (validation). PROCEED only if every clean update is accepted in every seed;
            otherwise STOP and report (no other schedule, no threshold change)
  run       refuses unless pilot says proceed. Per seed: five_f0.9_s<seed> (soft R1, registered method) and
            five_f1.0_s<seed> (threeway+relabel R1, the FU2 method). Arms clean / poisoned / poisoned_repair.
            Validation only; writes five/summary.json
  score-test  deployed models on held-out test, ONCE; refuses unless five/summary.json is byte-identical to
            experiments/results/12_five/summary.json in --decisions-commit

Usage: python s16_five.py prepare|pilot|run --src <nids_out of notebook 04> --work /kaggle/working/nids_out
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import s3_continual as S3
from common import load_json, save_json

HERE = Path(__file__).resolve().parent
TAG = "five"
SCHEDULE = [["Syn"], ["LDAP"], ["UDP"], ["TFTP"]]
ATTACK_CAP = 13890
SEEDS = (42, 43, 44)
ARMS = ("clean", "poisoned", "poisoned_repair")
COMMITTED = "experiments/results/12_five"
STAGE1 = ["split_manifest.json", "class_map.json", "features.json", "clean_train.parquet", "clean_calib.parquet",
          "clean_val.parquet", "clean_test.parquet"]
METHODS = {0.9: [], 1.0: ["--repair-rule", "threeway", "--relabel-to-teacher"]}


def call(*args):
    cmd = [sys.executable, *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def name(f, s):
    return f"five_f{f}_s{s}"


def s3(w, f, s, arms, device):
    call(HERE / "s3_continual.py", "--out", w, "--tag", TAG, "--scenario", "cii", "--run", name(f, s), "--seed", s,
         "--poison-target", "Syn", "--poison-to", "BENIGN", "--poison-frac", f, *METHODS[f], "--defer-test",
         "--arms", ",".join(arms), "--device", device, "--resume")


def prepare(a):
    src, w = Path(a.src), Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    for f in STAGE1:
        if not (w / f).exists() and (src / f).exists():
            shutil.copy2(src / f, w / f)
    if (w / TAG / "cii" / "experience_manifest.json").exists():
        print("Stage 1b five: done earlier")
        return
    sched = w / TAG / "schedule.json"
    sched.parent.mkdir(parents=True, exist_ok=True)
    save_json({"experiences": SCHEDULE}, sched)
    call(HERE / "s1b_experiences.py", "--out", w, "--schedule", sched, "--scenario", "cii", "--tag", TAG,
         "--attack-cap", ATTACK_CAP)
    m = load_json(w / TAG / "cii" / "experience_manifest.json")
    print("rows per experience:", m["counts_per_experience"])


def pilot(a):
    w = Path(a.work)
    out = w / TAG / "pilot.json"
    if out.exists():
        print("pilot decided earlier:", load_json(out)["proceed"])
        return load_json(out)["proceed"]
    per = {}
    for s in SEEDS:
        s3(w, 0.9, s, ["clean"], a.device)
        recs = load_json(w / "runs" / "cii" / name(0.9, s) / "clean" / "rounds.json")
        per[str(s)] = [{"E": r["experience"], "accepted": r["accepted"], "reasons": r["gate_reasons"]} for r in recs]
    proceed = all(all(x["accepted"] for x in v) for v in per.values())
    save_json({"proceed": proceed, "rule": "proceed only if every clean update is accepted in seeds 42-44",
               "per_seed": per, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, out)
    print("pilot PROCEED =", proceed)
    return proceed


def run(a):
    w = Path(a.work)
    if not pilot(a):
        raise SystemExit("STOP: the clean five-class pilot did not pass; report this (no fallback)")
    for s in SEEDS:
        for f in (0.9, 1.0):
            s3(w, f, s, list(ARMS), a.device)
    summ = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "schedule": SCHEDULE, "runs": []}
    for s in SEEDS:
        for f in (0.9, 1.0):
            rdir = w / "runs" / "cii" / name(f, s)
            e = {"name": name(f, s), "seed": s, "frac": f, "repair": "soft" if f == 0.9 else "threeway+relabel"}
            for arm in ARMS:
                rounds = load_json(rdir / arm / "rounds.json")
                n_up = len(SCHEDULE) - 1
                ck = rdir / arm / f"A{n_up}.pt"
                e[arm] = {"final_ckpt": str(ck.relative_to(w)), "final_sha256": S3.file_hash(ck),
                          "updates_accepted": len({x["experience"] for x in rounds if x["accepted"]}),
                          "updates": n_up, "r1_ran": sum(x["attempt"] == "R1" for x in rounds),
                          "r1_accepted": sum(x["attempt"] == "R1" and x["accepted"] for x in rounds),
                          "attempts": [{"E": x["experience"], "attempt": x["attempt"], "accepted": x["accepted"],
                                        "reasons": x["gate_reasons"], "val_recall": x["val"]["recall"],
                                        "benign_far": x["val"]["benign_far"],
                                        "n_relabelled": (x["teacher_weights"] or {}).get("n_relabelled")}
                                       for x in rounds]}
            summ["runs"].append(e)
    save_json(summ, w / TAG / "summary.json")
    for e in summ["runs"]:
        print(e["name"], {arm: f"{e[arm]['updates_accepted']}/{e[arm]['updates']}, R1 "
                               f"{e[arm]['r1_accepted']}/{e[arm]['r1_ran']}" for arm in ARMS})


def score_test(a):
    w = Path(a.work)
    out = w / TAG / "test.json"
    if out.exists():
        raise SystemExit(f"refuse: {out} exists; test is scored once")
    mine = (w / TAG / "summary.json").read_bytes()
    repo = Path(a.repo or HERE)
    got = subprocess.run(["git", "-C", str(repo), "show", f"{a.decisions_commit}:{COMMITTED}/summary.json"],
                         capture_output=True)
    assert got.returncode == 0 and got.stdout == mine, "refuse: five/summary.json differs from the committed decisions"
    summ = json.loads(mine)
    ctx, *_ = S3.load_context(w, TAG, "cii", a.device)
    res = {"decisions_commit": a.decisions_commit, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "runs": []}
    for e in summ["runs"]:
        hp = argparse.Namespace(**load_json(w / "runs" / "cii" / e["name"] / "config.json")["args"])
        row = {k: e[k] for k in ("name", "seed", "frac", "repair")}
        for arm in ARMS:
            ck = w / e[arm]["final_ckpt"]
            assert S3.file_hash(ck) == e[arm]["final_sha256"], f"refuse: {ck} is not the validated model"
            m, cls = S3.load_ckpt(ck, ctx, hp)
            met = S3.evaluate(m, cls, ctx["test"], torch.device(a.device), scope=ctx["scope"])
            row[arm] = {k: met[k] for k in ("n_rows", "rows_sha256", "macro_recall", "recall", "mean_conf",
                                             "benign_far", "n_benign")}
            row[arm]["classes"] = cls
        res["runs"].append(row)
    save_json(res, out)
    for r in res["runs"]:
        print(r["name"], {arm: {k: round(v, 4) for k, v in r[arm]["recall"].items()} for arm in ARMS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "pilot", "run", "score-test"])
    ap.add_argument("--src")
    ap.add_argument("--work", default="/kaggle/working/nids_out")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"prepare": prepare, "pilot": pilot, "run": run, "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
