"""RF (POST HOC): repeated small replay corruption under fixed-class TRAFFIC REFRESH. Separate from RC and RCS.

Pre-registration: experiments/prereg_refresh.md (committed before this code). RC and RCS stopped at their clean
stage (class-incremental schedules were not stable without poison). RF tests a different setting: E0 = BENIGN + Syn +
TFTP, then 6 refresh updates of fresh disjoint BENIGN + TFTP rows; no new class and no fresh Syn rows ever. It
tests traffic refresh with fixed classes, NOT new-class arrival.

It reuses s12's summarize / bundle / restore / score-test code with its own layout: work folder rf/, run names
rf_ref6_r<rho>_s<seed>, committed copies under experiments/results/10_rf/.

  prepare     copy + hash-check the Stage 1 files (as s12) and build Stage 1b of tag ref6 (s1b --allow-repeat)
  pilot       strict clean pilot: clean arm only, seeds 42-44, validation only (inside the rho 0.10 run folders).
              Proceed only if every seed satisfies the stability predicate; else STOP (no fallback, no other
              schedule). Writes rf/pilot.json with the clean files' sha256 even on failure; resumable
  run         refuses unless pilot.json says proceed. rho 0.10 (resumes the pilot runs: their clean arm is the
              control, not retrained), then rho 0.05 unless the measured T4 time > 90 min. Resumable
  summarize   checks the pilot's clean files are unchanged, then s12's trajectories and predictions
  zip-val / bundle / restore / score-test   as s12 (score-test checks experiments/results/10_rf/ in a git commit)

Stability predicate (per clean run, s13_rcs.stability): (a) every C1 accepted; (b) every C1's smallest gate margin
>= 0.01 (old Syn, old TFTP, benign FAR; no new-class terms); (c) max_t |Syn val recall(A_t) - A0| <= 0.02.

Usage (see notebooks/10_refresh_corruption.ipynb):
  python s14_refresh.py prepare --src <nids_out of notebook 04> --work /kaggle/working/nids_out
"""
from __future__ import annotations

import argparse
import shlex
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

import s12_repeated as S12
import s13_rcs as S13
import s3_continual as S3
from common import load_json, save_json

TAG = "ref6"
REFRESH_CLASS, N_UPDATES = "TFTP", 6  # E0 = BENIGN + Syn + TFTP; E1..E6 = fresh BENIGN + TFTP
SEEDS = (42, 43, 44)
MIN_MARGIN, SYN_BAND = 0.01, 0.02
# protected val rows of ref6 (s14 prepare on a local copy of the real rows, 2026-10-28, after the registration and
# before any RF run; no training)
REGISTERED_VAL = {"BENIGN": 9507, "Syn": 5973, "TFTP": 5991}
COMMITTED = "experiments/results/10_rf"  # rf_val.zip's contents (paths start with rf/, runs/ or ref6/)


def run_name(tag, rho, seed):
    return f"rf_{tag}_r{rho:.2f}_s{seed}"


def rcdir(a) -> Path:
    d = S12.work(a) / "rf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def schedule(a) -> list[list[str]]:
    return [["Syn", a.refresh_class]] + [[a.refresh_class]] * a.updates


def pilot_json(a) -> dict | None:
    f = rcdir(a) / "pilot.json"
    return load_json(f) if f.exists() else None


def runs_done(a):
    """(pilot.json, finished runs) in s12's format, so s12's summarize / bundle / score-test work unchanged."""
    p = pilot_json(a)
    out = []
    if p and p["proceed"]:
        for rho in S12.RHOS:
            for seed in a.seed_list:
                rdir = S12.work(a) / "runs" / "cii" / run_name(p["tag"], rho, seed)
                if S12.finished(rdir):
                    out.append((rho, seed, rdir))
    return p, out


def val_files(a) -> list[Path]:
    """The validation decision files (no buffers, no checkpoints): pilot decision, schedule, Stage 1b manifest,
    the clean files of the pilot runs, and s12's per-run files of the finished runs."""
    w, r = S12.work(a), rcdir(a)
    files = [r / f for f in ("pilot.json", "summary.json", "trajectory_val.csv", "diag_val.csv", "stopped.json")
             if (r / f).exists()]
    files += sorted(r.glob("schedule_*.json"))
    files += [w / a.tag / "cii" / "experience_manifest.json"]
    full = {rd.name for _, _, rd in runs_done(a)[1]}
    for rd in sorted((w / "runs" / "cii").glob("rf_*")):
        if rd.name in full:
            files += [rd / f for f in ("config.json", "run_manifest.json", "A0_metrics.json")]
            files += [rd / arm / f for arm in S12.ARMS for f in ("rounds.json", "val_done.json")]
            files += [rd / "poisoned" / "r1_diag.json"]
        else:
            files += [rd / f for f in ("config.json", "run_manifest.json", *S13.CLEAN_FILES[1:]) if (rd / f).exists()]
    return files


def use_rf_layout(a):
    """Point s12's shared code at the RF layout (s12 looks these names up at call time)."""
    S12.rcdir, S12.run_name, S12.runs_done, S12.val_files, S12.COMMITTED = rcdir, run_name, runs_done, val_files, COMMITTED


# ---- prepare --------------------------------------------------------------------
def stage1b(a):
    """Stage 1b of the refresh schedule (s1b --allow-repeat). Skipped only if all its files exist; in a restore
    session (manifest restored, rows not) it is rebuilt and must reproduce the restored manifest byte for byte."""
    w, want = S12.work(a), schedule(a)
    em = w / a.tag / "cii" / "experience_manifest.json"
    parts = [em, w / a.tag / "cii" / "experience_rids.npz", w / a.tag / "cii" / "scaler.json",
             w / a.tag / "protected_exclusions.json"]
    if all(f.exists() for f in parts):
        assert load_json(em)["schedule"] == want, f"refuse: {em} holds a different schedule"
        print(f"Stage 1b {a.tag}: done earlier, skipped")
    else:
        before = em.read_bytes() if em.exists() else None
        sched = rcdir(a) / f"schedule_{a.tag}.json"
        save_json({"experiences": want}, sched)
        S12.call("s1b_experiences.py", "--out", w, "--schedule", sched, "--scenario", "cii", "--tag", a.tag,
                 "--attack-cap", a.attack_cap, "--allow-repeat", *shlex.split(a.s1b_args or ""),
                 timing=rcdir(a) / "timing.json", label=f"s1b {a.tag}")
        assert before is None or em.read_bytes() == before, f"refuse: rebuilt {em} differs from the restored one"
    m = load_json(em)
    val = m["protected_exclusions"]["kept_by_split_class"]["val"]
    if registered(a):
        assert val == REGISTERED_VAL, f"refuse: protected val rows {val} differ from the registered {REGISTERED_VAL}"
        print("protected val rows equal the recorded feasibility check")
    print(pd.DataFrame(m["counts_per_experience"]).fillna(0).astype(int).to_string())
    return m


def registered(a) -> bool:
    return (a.tag, a.refresh_class, a.updates, a.attack_cap) == (TAG, REFRESH_CLASS, N_UPDATES, S12.ATTACK_CAP)


def prepare(a):
    S12.copy_stage1(Path(a.src), S12.work(a))
    stage1b(a)


# ---- strict clean pilot ---------------------------------------------------------
def pilot(a):
    out = rcdir(a) / "pilot.json"
    if out.exists():
        p = load_json(out)
        print(f"pilot decided earlier: PROCEED = {p['proceed']} ({p['reason']})")
        return
    stage1b(a)
    w = S12.work(a)
    per, hashes = {}, {}
    for seed in a.seed_list:  # clean arm only, inside the rho 0.10 run folder (validation only, no test)
        name = run_name(a.tag, S12.RHOS[0], seed)
        S12.call("s3_continual.py", "--out", w, "--tag", a.tag, "--run", name, "--seed", seed,
                 *S12.s3_flags(a, S12.RHOS[0]), "--arms", "clean", "--resume",
                 timing=rcdir(a) / "timing.json", label=f"pilot {name}")
        rdir = w / "runs" / "cii" / name
        per[str(seed)] = S13.stability(rdir, a.updates, a)
        hashes.update({str((rdir / f).relative_to(w)): S3.file_hash(rdir / f) for f in S13.CLEAN_FILES})
    n = sum(v["stable"] for v in per.values())
    proceed = n == len(a.seed_list)
    save_json({"proceed": proceed, "tag": a.tag, "classes": schedule(a)[1:],
               "schedule": schedule(a), "setting": "fixed-class traffic refresh (no new class after E0)",
               "seeds": list(a.seed_list), "reason": f"stable in {n}/{len(a.seed_list)} seeds", "per_seed": per,
               "clean_files_sha256": hashes,
               "rule": (f"stable run: every C1 accepted, every C1's smallest gate margin >= {a.min_margin}, "
                        f"max |Syn val recall(A_t) - A0| <= {a.syn_band}; proceed only if every seed is stable; "
                        "otherwise stop (no fallback, no other schedule)"),
               "min_margin": a.min_margin, "syn_band": a.syn_band,
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, out)
    print(f"pilot: PROCEED = {proceed} ({n}/{len(a.seed_list)} seeds stable)")


# ---- poisoned runs ----------------------------------------------------------------
def run(a):
    p = pilot_json(a)
    if not p or not p["proceed"]:
        raise SystemExit("refuse: no pilot.json with proceed = true (the strict clean pilot decides; do not change "
                         "the design)")
    timing = rcdir(a) / "timing.json"
    for i, rho in enumerate(S12.RHOS):
        total = load_json(timing)["total_seconds"]
        if i > 0 and total > S12.T4_CAP_S:
            save_json({"stopped_before_rho": rho, "total_seconds": total, "rule": "stopping rule 2: 90 min of T4"},
                      rcdir(a) / "stopped.json")
            print(f"stopping rule 2: {total / 60:.0f} min > 90 min after rho {S12.RHOS[i - 1]}; rho {rho} not run")
            return
        for seed in a.seed_list:
            name = run_name(p["tag"], rho, seed)
            if S12.finished(S12.work(a) / "runs" / "cii" / name):
                print(f"{name}: finished earlier, skipped")
                continue
            S12.call("s3_continual.py", "--out", S12.work(a), "--tag", p["tag"], "--run", name, "--seed", seed,
                     *S12.s3_flags(a, rho), "--arms", ",".join(S12.ARMS), "--resume", timing=timing,
                     label=f"run {name}")
    print(f"all runs done; measured time {load_json(timing)['total_seconds'] / 60:.1f} min")


def summarize(a):
    p = pilot_json(a)
    w = S12.work(a)
    bad = [f for f, h in p["clean_files_sha256"].items() if S3.file_hash(w / f) != h]
    assert not bad, f"refuse: the pilot's clean files changed after pilot.json: {bad}"
    S12.summarize(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "pilot", "run", "summarize", "zip-val", "bundle", "restore",
                                    "score-test"])
    ap.add_argument("--src", help="nids_out of notebook 04: clean_*.parquet and runs/cii/all12_cii_s42 (read only)")
    ap.add_argument("--work", default=None, help="writable nids_out for Stage 1b, runs and rf/ (default: work dir)")
    ap.add_argument("--attack-cap", type=int, default=S12.ATTACK_CAP)
    ap.add_argument("--refresh-class", default=REFRESH_CLASS, help="smoke test only (default = registered)")
    ap.add_argument("--updates", type=int, default=N_UPDATES, help="smoke test only (default = registered)")
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN, help="smoke test only")
    ap.add_argument("--syn-band", type=float, default=SYN_BAND, help="smoke test only")
    ap.add_argument("--s1b-args", default="", help="smoke test only (synthetic data)")
    ap.add_argument("--s3-args", default="", help="smoke test only (synthetic data)")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None, help="git checkout holding the decisions commit (default: this code's)")
    ap.add_argument("--bundle", default=None, help="zip-val / bundle: zip to write; restore: zip or extracted folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.seed_list, a.tag = list(SEEDS), TAG
    use_rf_layout(a)
    {"prepare": prepare, "pilot": pilot, "run": run, "summarize": summarize, "zip-val": S12.zip_val,
     "bundle": S12.bundle, "restore": S12.restore, "score-test": S12.score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
