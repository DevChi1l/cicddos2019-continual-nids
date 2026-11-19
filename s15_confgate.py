"""CG (POST HOC, 2026-11-19): a confidence-aware gate. Pre-registration: experiments/prereg_confgate.md.

Main and RF showed that the recall gate cannot see a poisoned update that keeps the argmax but lowers confidence
(mean p(Syn) 0.994 -> ~0.79 in main; 0.99 -> 0.77 in RF). CG re-runs those two settings with ONE change: s3's
existing --gate-conf-drop 0.02, i.e. a candidate is also rejected if, for any old attack class k,
mean p(k | val rows of k) falls more than 0.02 below the last accepted model. Everything else is the original
recipe (s3 defaults; RF flags for the refresh setting). Soft R1 is the registered repair.

  CG-RF    tag ref6 (E0 BENIGN+Syn+TFTP, 6 TFTP refresh updates), rho 0.10 and 0.05, seeds 42-44,
           poison-every-update, arms clean / poisoned / poisoned_repair
  CG-MAIN  four-class CII (BENIGN+Syn, +LDAP, +UDP), f = 0.5, seeds 42-44, same three arms

  run         validation only (s3 --defer-test). Writes cg/summary.json. Resumable.
  score-test  each deployed model on held-out test, ONCE. Refuses unless cg/summary.json is byte-identical to
              experiments/results/11_confgate/summary.json in --decisions-commit, and the models have the hashes
              recorded there.

Usage (Kaggle T4, notebook 12 or the CG cell):
  python s15_confgate.py run --src <nids_out of notebook 04> --work /kaggle/working/nids_out
  python s15_confgate.py score-test --work /kaggle/working/nids_out --decisions-commit <sha>
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
CONF_DROP = 0.02  # fixed before any CG run: about 2x the largest clean drop in the committed logs (0.0108)
SEEDS = (42, 43, 44)
RHOS = (0.10, 0.05)
ARMS = ("clean", "poisoned", "poisoned_repair")
COMMITTED = "experiments/results/11_confgate"
STAGE1 = ["split_manifest.json", "class_map.json", "features.json", "clean_train.parquet", "clean_calib.parquet",
          "clean_val.parquet", "clean_test.parquet"]


def runs():
    out = []
    for rho in RHOS:
        for s in SEEDS:
            out.append({"name": f"cg_ref6_r{rho:.2f}_s{s}", "setting": "CG-RF", "tag": "ref6", "seed": s,
                        "frac": rho, "extra": ["--poison-every-update"]})
    for s in SEEDS:
        out.append({"name": f"cg_main_f0.5_s{s}", "setting": "CG-MAIN", "tag": "", "seed": s, "frac": 0.5,
                    "extra": []})
    return out


def call(*args):
    cmd = [sys.executable, *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def prepare(src: Path, w: Path):
    w.mkdir(parents=True, exist_ok=True)
    for f in STAGE1 + ["protected_exclusions.json"]:
        if not (w / f).exists() and (src / f).exists():
            shutil.copy2(src / f, w / f)
    if not (w / "cii").exists():
        shutil.copytree(src / "cii", w / "cii")
    if not (w / "ref6" / "cii" / "experience_manifest.json").exists():
        call(HERE / "s14_refresh.py", "prepare", "--src", src, "--work", w)


def finished(rdir: Path, n_updates: int) -> bool:
    return all((rdir / arm / f"A{n_updates}.pt").exists() and (rdir / arm / "rounds.json").exists() for arm in ARMS)


def run(a):
    src, w = Path(a.src), Path(a.work)
    prepare(src, w)
    t0 = time.time()
    for r in runs():
        rdir = w / "runs" / "cii" / r["name"]
        n_up = len(load_json(w / r["tag"] / "cii" / "experience_manifest.json")["schedule"]) - 1 if r["tag"] else \
            len(load_json(w / "cii" / "experience_manifest.json")["schedule"]) - 1
        if finished(rdir, n_up):
            print(f"{r['name']}: finished earlier, skipped")
            continue
        call(HERE / "s3_continual.py", "--out", w, *(["--tag", r["tag"]] if r["tag"] else []),
             "--scenario", "cii", "--run", r["name"], "--seed", r["seed"], "--poison-target", "Syn",
             "--poison-to", "BENIGN", "--poison-frac", r["frac"], *r["extra"], "--defer-test",
             "--gate-conf-drop", CONF_DROP, "--arms", ",".join(ARMS), "--device", a.device, "--resume")
    summary = {"conf_drop": CONF_DROP, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "seconds": round(time.time() - t0, 1), "runs": []}
    for r in runs():
        rdir = w / "runs" / "cii" / r["name"]
        entry = {k: r[k] for k in ("name", "setting", "seed", "frac")}
        for arm in ARMS:
            rounds = load_json(rdir / arm / "rounds.json")
            n_up = max(x["experience"] for x in rounds)
            ck = rdir / arm / f"A{n_up}.pt"
            entry[arm] = {
                "final_ckpt": str(ck.relative_to(w)), "final_sha256": S3.file_hash(ck),
                "updates_accepted": len({x["experience"] for x in rounds if x["accepted"]}), "updates": n_up,
                "r1_ran": sum(x["attempt"] == "R1" for x in rounds),
                "r1_accepted": sum(x["attempt"] == "R1" and x["accepted"] for x in rounds),
                "attempts": [{"E": x["experience"], "attempt": x["attempt"], "accepted": x["accepted"],
                              "reasons": x["gate_reasons"], "Syn_val_recall": x["val"]["recall"].get("Syn"),
                              "Syn_val_mean_p": x["val"]["mean_conf"].get("Syn"),
                              "Syn_ref_mean_p": x["ref_val"]["mean_conf"].get("Syn"),
                              "n_replay_flipped": x["n_replay_flipped"],
                              "conf_check": x["conf_check"]} for x in rounds]}
        summary["runs"].append(entry)
    save_json(summary, w / "cg" / "summary.json")
    print(json.dumps([{"run": e["name"], **{arm: f"{e[arm]['updates_accepted']}/{e[arm]['updates']} acc, "
                                                 f"R1 {e[arm]['r1_accepted']}/{e[arm]['r1_ran']}" for arm in ARMS}}
                      for e in summary["runs"]], indent=1))


def score_test(a):
    w = Path(a.work)
    out = w / "cg" / "test.json"
    if out.exists():
        raise SystemExit(f"refuse: {out} exists; test is scored once")
    mine = (w / "cg" / "summary.json").read_bytes()
    repo = Path(a.repo or HERE)
    committed = subprocess.run(["git", "-C", str(repo), "show", f"{a.decisions_commit}:{COMMITTED}/summary.json"],
                               capture_output=True)
    assert committed.returncode == 0, f"refuse: {COMMITTED}/summary.json not found in {a.decisions_commit}"
    assert committed.stdout == mine, "refuse: cg/summary.json differs from the committed decisions"
    summ = json.loads(mine)
    res, ctxs = {"decisions_commit": a.decisions_commit,
                 "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": []}, {}
    for e in summ["runs"]:
        tag = "ref6" if e["setting"] == "CG-RF" else ""
        if tag not in ctxs:
            ctxs[tag], *_ = S3.load_context(w, tag, "cii", a.device)
        ctx = ctxs[tag]
        hp = argparse.Namespace(**load_json(w / "runs" / "cii" / e["name"] / "config.json")["args"])
        row = {"name": e["name"], "setting": e["setting"], "seed": e["seed"], "frac": e["frac"]}
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
        print(r["name"], {arm: (round(r[arm]["recall"].get("Syn", 0), 4), round(r[arm]["mean_conf"].get("Syn", 0), 4),
                                {k: round(v, 4) for k, v in r[arm]["recall"].items()}) for arm in ARMS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score-test"])
    ap.add_argument("--src", help="nids_out of notebook 04 (Stage 1 files and cii/), read only")
    ap.add_argument("--work", default="/kaggle/working/nids_out")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"run": run, "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
