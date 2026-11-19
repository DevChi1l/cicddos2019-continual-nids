"""SEEDS (2026-11-19): five more seeds (45-49) for the main defended results. Pre-registration:
experiments/prereg_seeds.md. Same code paths as the sweep, FU2 and FIVE; nothing is tuned.

  4-class (tag "", CII; the sweep / FU2 setting):
    s4_f{0.75,0.9,0.95}_s{seed}   soft R1 (the registered method)            arms clean / poisoned / poisoned_repair
    s4_f1.0_s{seed}               threeway+relabel R1 (the FU2 method)        arms clean / poisoned / poisoned_repair
  5-class (tag "five"; FIVE's schedule, its clean pilot passed in seeds 42-44):
    s5_f0.9_s{seed}               soft R1
    s5_f1.0_s{seed}               threeway+relabel R1

  run         validation only (s3 --defer-test); writes seeds/summary.json; resumable
  score-test  every deployed model on held-out test ONCE; refuses unless seeds/summary.json is byte-identical to
              experiments/results/13_seeds/summary.json in --decisions-commit and the model hashes match

Usage (Kaggle T4):
  python s17_seeds.py run --src <nids_out of notebook 04> --work /kaggle/working/nids_out
  python s17_seeds.py score-test --work /kaggle/working/nids_out --decisions-commit <sha>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import s3_continual as S3
from common import load_json, save_json

HERE = Path(__file__).resolve().parent
SEEDS = (45, 46, 47, 48, 49)
ARMS = ("clean", "poisoned", "poisoned_repair")
RELABEL = ["--repair-rule", "threeway", "--relabel-to-teacher"]
COMMITTED = "experiments/results/13_seeds"
STAGE1 = ["split_manifest.json", "class_map.json", "features.json", "clean_train.parquet", "clean_calib.parquet",
          "clean_val.parquet", "clean_test.parquet"]


def runs():
    out = []
    for s in SEEDS:
        for f in (0.75, 0.9, 0.95, 1.0):
            out.append({"name": f"s4_f{f}_s{s}", "classes": 4, "tag": "", "seed": s, "frac": f,
                        "repair": "threeway+relabel" if f == 1.0 else "soft", "extra": RELABEL if f == 1.0 else []})
        for f in (0.9, 1.0):
            out.append({"name": f"s5_f{f}_s{s}", "classes": 5, "tag": "five", "seed": s, "frac": f,
                        "repair": "threeway+relabel" if f == 1.0 else "soft", "extra": RELABEL if f == 1.0 else []})
    return out


def call(*args):
    cmd = [sys.executable, *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def prepare(src: Path, w: Path, device: str):
    w.mkdir(parents=True, exist_ok=True)
    for f in STAGE1 + ["protected_exclusions.json"]:
        if not (w / f).exists() and (src / f).exists():
            shutil.copy2(src / f, w / f)
    if not (w / "cii").exists():
        shutil.copytree(src / "cii", w / "cii")
    if not (w / "five" / "cii" / "experience_manifest.json").exists():
        call(HERE / "s16_five.py", "prepare", "--src", src, "--work", w, "--device", device)


def n_updates(w: Path, tag: str) -> int:
    return len(load_json(w / tag / "cii" / "experience_manifest.json")["schedule"]) - 1 if tag else \
        len(load_json(w / "cii" / "experience_manifest.json")["schedule"]) - 1


def run(a):
    src, w = Path(a.src), Path(a.work)
    prepare(src, w, a.device)
    t0 = time.time()
    for r in runs():
        rdir = w / "runs" / "cii" / r["name"]
        n = n_updates(w, r["tag"])
        if all((rdir / arm / f"A{n}.pt").exists() for arm in ARMS):
            print(f"{r['name']}: finished earlier, skipped")
            continue
        call(HERE / "s3_continual.py", "--out", w, *(["--tag", r["tag"]] if r["tag"] else []), "--scenario", "cii",
             "--run", r["name"], "--seed", r["seed"], "--poison-target", "Syn", "--poison-to", "BENIGN",
             "--poison-frac", r["frac"], *r["extra"], "--defer-test", "--arms", ",".join(ARMS),
             "--device", a.device, "--resume")
        print(f"elapsed {time.time() - t0:.0f}s", flush=True)
    summ = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seeds": list(SEEDS),
            "seconds": round(time.time() - t0, 1), "runs": []}
    for r in runs():
        rdir = w / "runs" / "cii" / r["name"]
        n = n_updates(w, r["tag"])
        e = {k: r[k] for k in ("name", "classes", "seed", "frac", "repair")}
        for arm in ARMS:
            rounds = load_json(rdir / arm / "rounds.json")
            ck = rdir / arm / f"A{n}.pt"
            e[arm] = {"final_ckpt": str(ck.relative_to(w)), "final_sha256": S3.file_hash(ck), "updates": n,
                      "updates_accepted": len({x["experience"] for x in rounds if x["accepted"]}),
                      "r1_ran": sum(x["attempt"] == "R1" for x in rounds),
                      "r1_accepted": sum(x["attempt"] == "R1" and x["accepted"] for x in rounds),
                      "attempts": [{"E": x["experience"], "attempt": x["attempt"], "accepted": x["accepted"],
                                    "reasons": x["gate_reasons"], "val_recall": x["val"]["recall"],
                                    "benign_far": x["val"]["benign_far"],
                                    "n_relabelled": (x["teacher_weights"] or {}).get("n_relabelled")}
                                   for x in rounds]}
        summ["runs"].append(e)
    save_json(summ, w / "seeds" / "summary.json")
    for e in summ["runs"]:
        print(e["name"], {arm: f"{e[arm]['updates_accepted']}/{e[arm]['updates']} R1 {e[arm]['r1_accepted']}/"
                               f"{e[arm]['r1_ran']}" for arm in ARMS})


def score_test(a):
    w = Path(a.work)
    out = w / "seeds" / "test.json"
    if out.exists():
        raise SystemExit(f"refuse: {out} exists; test is scored once")
    mine = (w / "seeds" / "summary.json").read_bytes()
    got = subprocess.run(["git", "-C", str(Path(a.repo or HERE)), "show",
                          f"{a.decisions_commit}:{COMMITTED}/summary.json"], capture_output=True)
    assert got.returncode == 0 and got.stdout == mine, "refuse: seeds/summary.json differs from the committed decisions"
    summ = json.loads(mine)
    ctxs = {}
    res = {"decisions_commit": a.decisions_commit, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "runs": []}
    for e in summ["runs"]:
        tag = "five" if e["classes"] == 5 else ""
        if tag not in ctxs:
            ctxs[tag], *_ = S3.load_context(w, tag, "cii", a.device)
        ctx = ctxs[tag]
        hp = argparse.Namespace(**load_json(w / "runs" / "cii" / e["name"] / "config.json")["args"])
        row = {k: e[k] for k in ("name", "classes", "seed", "frac", "repair")}
        for arm in ARMS:
            ck = w / e[arm]["final_ckpt"]
            assert S3.file_hash(ck) == e[arm]["final_sha256"], f"refuse: {ck} is not the validated model"
            m, cls = S3.load_ckpt(ck, ctx, hp)
            met = S3.evaluate(m, cls, ctx["test"], torch.device(a.device), scope=ctx["scope"])
            row[arm] = {k: met[k] for k in ("n_rows", "rows_sha256", "accuracy", "macro_recall", "recall", "mean_conf",
                                             "benign_far", "n_benign", "confusion")}
            row[arm]["classes"] = cls
        res["runs"].append(row)
    save_json(res, out)
    for r in res["runs"]:
        print(r["name"], {arm: {k: round(v, 4) for k, v in r[arm]["recall"].items()} for arm in ARMS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score-test"])
    ap.add_argument("--src")
    ap.add_argument("--work", default="/kaggle/working/nids_out")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"run": run, "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
