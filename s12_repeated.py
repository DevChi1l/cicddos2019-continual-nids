"""Repeated small replay corruption (RC): 5% / 10% more Syn -> BENIGN flips before EVERY update, 6 updates.

Pre-registration: experiments/prereg_repeated_corruption.md (committed before any RC code or run). This is NOT the
sweep (one large flip before E1): no sweep number is an RC result. Everything below follows the registration; the
"Clarifications before execution" section there fixes the few details it left open.

  prepare     copy the Stage 1 files of notebook 04's output into <work> (hash-checked against the all12 run),
              write the rep6 schedule and run Stage 1b (--tag rep6, CII); checks the registered protected val rows
  pilot       clean arm only, seeds 42-44, validation only (runs rc_<tag>_r0.10_s<seed> --arms clean). Proceed if
              the clean arm accepts every update in >= 2 of 3 seeds; else ONE registered fallback (drop classes whose
              clean C1 was rejected in >= 2 seeds, keep name order, new tag rep6fb) and the pilot once more; stop if it
              still fails or fewer than 5 updates remain. Writes rc/pilot.json; resumable
  run         rho 0.10 then 0.05, seeds 42-44, arms clean / poisoned / poisoned_repair (soft R1), forced soft-R1
              diagnostic on the poisoned arm, every update, validation only (s3 --defer-test). Stops before
              rho 0.05 if the measured T4 time already exceeds 90 min. Resumable (s3 --resume)
  summarize   trajectories and registered predictions from saved files -> rc/trajectory_val.csv, rc/diag_val.csv,
              rc/summary.json (validation only; also the sha256 of every deployed model A0..A6)
  bundle      deployed models + validation files + sha256 manifest (no test), for a later session
  restore     verify every hash in a bundle, then copy it into <work>
  score-test  every arm's deployed models A0..A6 on test ONCE; refuses unless the validation files are
              byte-identical to their copies under experiments/results/08_rc/ in the given git commit

Usage (see notebooks/08_repeated_corruption.ipynb):
  python s12_repeated.py prepare --src <nids_out of notebook 04> --work /kaggle/working/nids_out
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import s3_continual as S3
from common import default_work_dir, load_json, save_json

HERE = Path(__file__).resolve().parent
TAG, FALLBACK_TAG = "rep6", "rep6fb"
REGISTERED_CLASSES = ["LDAP", "MSSQL", "NTP", "SNMP", "SSDP", "TFTP"]  # E1..E6, name order; E0 = BENIGN + Syn
ATTACK_CAP = 9312
RHOS = (0.10, 0.05)  # registered order: the stronger case first
SEEDS = (42, 43, 44)
ARMS = ("clean", "poisoned", "poisoned_repair")
T4_CAP_S = 90 * 60
MIN_UPDATES = 5
# protected val rows of the registered schedule (Stage 1b on a scratch copy, 2026-10-27, before registration)
REGISTERED_VAL = {"BENIGN": 9507, "Syn": 5973, "LDAP": 2297, "MSSQL": 5592, "NTP": 5994, "SNMP": 4961,
                  "SSDP": 4719, "TFTP": 5994}
STAGE1 = ["split_manifest.json", "class_map.json", "features.json", "clean_train.parquet", "clean_calib.parquet",
          "clean_val.parquet", "clean_test.parquet"]
COMMITTED = "experiments/results/08_rc"  # rc_val.zip's contents (paths start with rc/ or runs/ or the tag)


def run_name(tag, rho, seed):
    return f"rc_{tag}_r{rho:.2f}_s{seed}"


def work(a) -> Path:
    return Path(a.work or default_work_dir())


def s3_flags(a, rho):
    """The registered s3 settings (s3 defaults = the all12 / sweep recipe and gate) plus the RC flags."""
    return ["--scenario", "cii", "--poison-target", "Syn", "--poison-to", "BENIGN", "--poison-frac", rho,
            "--poison-every-update", "--defer-test", "--forced-r1", "soft", "--forced-r1-arms", "poisoned",
            "--device", a.device, *shlex.split(a.s3_args or "")]


def call(script, *args, timing=None, label=""):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=HERE)
    if timing is not None:  # measured GPU-session time of every s3 / s1b call (stopping rule 3)
        tj = load_json(timing) if timing.exists() else {"calls": []}
        tj["calls"].append({"label": label, "seconds": time.time() - t0,
                            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        tj["total_seconds"] = sum(c["seconds"] for c in tj["calls"])
        save_json(tj, timing)


def rcdir(a) -> Path:
    d = work(a) / "rc"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- prepare ----------------------------------------------------------------
def stage1_ref(src: Path) -> dict:
    """The Stage 1 file hashes the committed all12 runs used (the same capped 12-label rows)."""
    return load_json(src / "runs" / "cii" / "all12_cii_s42" / "config.json")["inputs"]


def copy_stage1(src: Path, w: Path):
    ref = stage1_ref(src)
    w.mkdir(parents=True, exist_ok=True)
    for n in STAGE1:
        if src.resolve() != w.resolve() and not (w / n).exists():
            shutil.copy2(src / n, w / n)
        if n in ref:
            assert S3.file_hash(w / n) == ref[n], f"refuse: {n} differs from the file all12 used"
    print(f"Stage 1 files in {w}; {sum(n in ref for n in STAGE1)} of them hash-checked against all12_cii_s42")


def stage1b(a, tag, classes):
    """Stage 1b for one tag. Skipped only if all its files exist; in a restore session (manifest restored, rows
    not) it is rebuilt and must reproduce the restored experience_manifest.json byte for byte."""
    w = work(a)
    sched = rcdir(a) / f"schedule_{tag}.json"
    want = [["Syn"]] + [[k] for k in classes]
    em = w / tag / "cii" / "experience_manifest.json"
    parts = [em, w / tag / "cii" / "experience_rids.npz", w / tag / "cii" / "scaler.json",
             w / tag / "protected_exclusions.json"]
    if all(f.exists() for f in parts):
        assert load_json(em)["schedule"] == want, f"refuse: {em} holds a different schedule"
        print(f"Stage 1b {tag}: done earlier, skipped")
    else:
        before = em.read_bytes() if em.exists() else None
        save_json({"experiences": want}, sched)
        call("s1b_experiences.py", "--out", w, "--schedule", sched, "--scenario", "cii", "--tag", tag,
             "--attack-cap", a.attack_cap, *shlex.split(a.s1b_args or ""),
             timing=rcdir(a) / "timing.json", label=f"s1b {tag}")
        assert before is None or em.read_bytes() == before, f"refuse: rebuilt {em} differs from the restored one"
    m = load_json(em)
    val = m["protected_exclusions"]["kept_by_split_class"]["val"]
    if classes == REGISTERED_CLASSES and a.attack_cap == ATTACK_CAP:
        assert val == REGISTERED_VAL, f"refuse: protected val rows {val} differ from the registered {REGISTERED_VAL}"
        print("protected val rows equal the registered feasibility check")
    print(pd.DataFrame(m["counts_per_experience"]).fillna(0).astype(int).to_string())
    return m


def prepare(a):
    """Stage 1 copy + Stage 1b of the registered tag, and of a fallback tag if a (restored) pilot used one."""
    copy_stage1(Path(a.src), work(a))
    pj = work(a) / "rc" / "pilot.json"
    tags = load_json(pj)["attempts"] if pj.exists() else [{"tag": TAG, "classes": a.class_list}]
    for x in tags:
        stage1b(a, x["tag"], x["classes"])


# ---- pilot ------------------------------------------------------------------
def clean_pilot(a, tag, n_updates):
    rows = {}
    for seed in a.seed_list:
        name = run_name(tag, RHOS[0], seed)
        call("s3_continual.py", "--out", work(a), "--tag", tag, "--run", name, "--seed", seed, *s3_flags(a, RHOS[0]),
             "--arms", "clean", "--resume", timing=rcdir(a) / "timing.json", label=f"pilot {name}")
        recs = load_json(work(a) / "runs" / "cii" / name / "clean" / "rounds.json")
        assert len(recs) == n_updates
        rows[seed] = {r["new_classes"][0]: bool(r["accepted"]) for r in recs}
    return rows


def pilot(a):
    out = rcdir(a) / "pilot.json"
    if out.exists():
        p = load_json(out)
        print(f"pilot decided earlier: proceed={p['proceed']} tag={p['tag']} ({p['reason']})")
        return
    attempts, tag, classes = [], TAG, list(a.class_list)
    need = int(np.ceil(2 * len(a.seed_list) / 3))  # >= 2 of 3 seeds
    while True:
        stage1b(a, tag, classes)
        acc = clean_pilot(a, tag, len(classes))
        all_ok = sum(all(v.values()) for v in acc.values())
        rejected = {k: sum(not acc[s][k] for s in acc) for k in classes}
        attempts.append({"tag": tag, "classes": classes, "accepted_by_seed": {str(s): v for s, v in acc.items()},
                         "seeds_accepting_all": all_ok, "rejections_per_class": rejected})
        if all_ok >= need:
            proceed, reason = True, f"clean arm accepted all {len(classes)} updates in {all_ok}/{len(acc)} seeds"
            break
        if tag == FALLBACK_TAG:
            proceed, reason = False, f"fallback pilot also failed ({all_ok}/{len(acc)} seeds accept all): stop and report"
            break
        drop = [k for k, n in rejected.items() if n >= need]
        classes = [k for k in classes if k not in drop]
        if len(classes) < MIN_UPDATES or not drop:
            proceed, reason = False, (f"pilot failed; fallback would leave {len(classes)} updates (< {MIN_UPDATES})"
                                      if drop else "pilot failed but no class was rejected in >= 2 seeds")
            break
        print(f"pilot failed ({all_ok}/{len(acc)} seeds): registered fallback drops {drop}, reruns once")
        tag = FALLBACK_TAG
    save_json({"proceed": proceed, "tag": tag, "classes": classes, "reason": reason, "attempts": attempts,
               "rule": "proceed if the clean arm accepts every update in >= 2 of 3 seeds; otherwise one fallback "
                       "(drop classes whose clean C1 was rejected in >= 2 seeds, keep name order) and the pilot once "
                       "more; stop if it still fails or fewer than 5 updates remain"}, out)
    print(f"pilot: proceed={proceed} tag={tag} ({reason})")


# ---- run ----------------------------------------------------------------------
def finished(rdir: Path) -> bool:
    return (rdir / "run_manifest.json").exists() and all((rdir / arm / "val_done.json").exists() for arm in ARMS)


def run(a):
    p = load_json(rcdir(a) / "pilot.json")
    if not p["proceed"]:
        raise SystemExit(f"refuse: the pilot said stop ({p['reason']})")
    timing = rcdir(a) / "timing.json"
    for i, rho in enumerate(RHOS):
        total = load_json(timing)["total_seconds"] if timing.exists() else 0.0
        if i > 0 and total > T4_CAP_S:
            save_json({"stopped_before_rho": rho, "total_seconds": total, "rule": "stopping rule 3: 90 min of T4"},
                      rcdir(a) / "stopped.json")
            print(f"stopping rule 3: {total / 60:.0f} min > 90 min after rho {RHOS[i - 1]}; rho {rho} not run")
            return
        for seed in a.seed_list:
            name = run_name(p["tag"], rho, seed)
            if finished(work(a) / "runs" / "cii" / name):
                print(f"{name}: finished earlier, skipped")
                continue
            call("s3_continual.py", "--out", work(a), "--tag", p["tag"], "--run", name, "--seed", seed,
                 *s3_flags(a, rho), "--arms", ",".join(ARMS), "--resume", timing=timing, label=f"run {name}")
    print(f"all runs done; measured time {load_json(timing)['total_seconds'] / 60:.1f} min")


# ---- summarize ------------------------------------------------------------------
def margins(rec, a):
    """Gate margins of one attempt (>= 0 passes) and the binding (smallest) one."""
    v, r = rec["val"], rec["ref_val"]
    m = {f"old {k}": v["recall"].get(k, 0.0) - (r["recall"][k] - a.gate_old_drop) for k in rec["old_classes"]
         if k != "BENIGN"}
    m.update({f"new {k}": v["recall"].get(k, 0.0) - a.gate_new_min for k in rec["new_classes"]})
    m["benign FAR"] = a.gate_far_max - v["benign_far"]
    m["new-class FAR"] = a.gate_newfar_max - v["new_class_far"]
    k = min(m, key=m.get)
    return m, k


def runs_done(a):
    p = load_json(rcdir(a) / "pilot.json")
    out = []
    for rho in RHOS:
        for seed in a.seed_list:
            rdir = work(a) / "runs" / "cii" / run_name(p["tag"], rho, seed)
            if finished(rdir):
                out.append((rho, seed, rdir))
    return p, out


def model_hashes(path: Path, ctx, hp):
    return S3.file_hash(path), S3.state_hash(S3.load_ckpt(path, ctx, hp)[0])


def summarize(a):
    p, runs = runs_done(a)
    assert runs, "no finished run"
    ctx = S3.load_context(work(a), p["tag"], "cii", "cpu")[0]
    traj, diag, summ = [], [], {"tag": p["tag"], "classes": p["classes"], "runs": {}}
    for rho, seed, rdir in runs:
        cfg, man = load_json(rdir / "config.json"), load_json(rdir / "run_manifest.json")
        hp = Namespace(**cfg["args"])
        assert man["T_unchanged"] and "A0_test" not in man, f"{rdir.name}: teacher changed or test scored"
        a0 = load_json(rdir / "A0_metrics.json")["val"]
        b1 = pd.read_csv(rdir / "clean" / "buffer_before_E1.csv")
        n_syn = int((b1["true_label"] == "Syn").sum())  # stored Syn rows (2,000 on real data)
        n0 = int(round(rho * n_syn))  # s3's per-update draw
        rs = {"rho": rho, "seed": seed, "config_args": cfg["args"], "T_sha256": man["T_sha256"], "arms": {}}
        rs["A0"] = dict(zip(("file_sha256", "state_sha256"), model_hashes(rdir / "A0.pt", ctx, hp)))
        flips = {}
        for arm in ARMS:
            recs = load_json(rdir / arm / "rounds.json")
            assert not (rdir / arm / "final_test.json").exists(), f"{rdir.name}/{arm}: test scored"
            ad = {"deployed": [rs["A0"]], "decisions": []}
            row0 = {"run": rdir.name, "rho": rho, "seed": seed, "arm": arm, "t": 0, "new_class": "",
                    "flipped_share": 0.0, "c1_accepted": None, "r1_ran": False, "r1_accepted": None,
                    "deployed_from": "A0", "Syn_recall": a0["recall"]["Syn"], "Syn_mean_p": a0["mean_conf"]["Syn"],
                    "benign_far": a0["benign_far"], "n_learned": len(a0["recall"]),
                    "recall": json.dumps(a0["recall"])}
            traj.append(row0)
            for t in range(1, len(p["classes"]) + 1):
                att = {r["attempt"]: r for r in recs if r["experience"] == t}
                c1, r1 = att["C1"], att.get("R1")
                acc = next((r for r in (c1, r1) if r is not None and r["accepted"]), None)
                v = acc["val"] if acc else c1["ref_val"]  # rejected -> the previous model stays deployed
                m, bind = margins(c1, hp)
                buf = pd.read_csv(rdir / arm / f"buffer_before_E{t}.csv")
                nfl = int(buf["flipped"].sum())
                assert nfl == c1["n_replay_flipped"] == (n0 * t if arm != "clean" else 0), (rdir.name, arm, t, nfl)
                if arm != "clean":
                    flips.setdefault(t, {})[arm] = sorted(buf.loc[buf["flipped"], "rid"].tolist())
                    assert (buf.loc[buf["flipped"], "true_label"] == "Syn").all()
                traj.append({**row0, "t": t, "new_class": c1["new_classes"][0],
                             "flipped_share": nfl / n_syn, "c1_accepted": c1["accepted"],
                             "r1_ran": r1 is not None, "r1_accepted": None if r1 is None else r1["accepted"],
                             "deployed_from": acc["attempt"] if acc else "kept", "Syn_recall": v["recall"]["Syn"],
                             "Syn_mean_p": v["mean_conf"]["Syn"], "benign_far": v["benign_far"],
                             "n_learned": len(v["recall"]), "recall": json.dumps(v["recall"]),
                             "c1_Syn_recall": c1["val"]["recall"].get("Syn", 0.0),
                             "c1_Syn_floor": c1["ref_val"]["recall"]["Syn"] - hp.gate_old_drop,
                             "c1_Syn_margin": m["old Syn"], "c1_binding": bind, "c1_min_margin": m[bind],
                             "c1_Syn_mean_p": c1["val"]["mean_conf"].get("Syn", 0.0),
                             "c1_gate_reasons": "; ".join(c1["gate_reasons"])})
                ad["deployed"].append(dict(zip(("file_sha256", "state_sha256"),
                                               model_hashes(rdir / arm / f"A{t}.pt", ctx, hp))))
                ad["decisions"].append(traj[-1]["deployed_from"])
            if arm == "poisoned":
                for d in load_json(rdir / arm / "r1_diag.json"):
                    tw = d["teacher_weights"]
                    diag.append({"run": rdir.name, "rho": rho, "seed": seed, "t": d["experience"],
                                 "method": d["method"], "would_pass_gate": d["would_pass_gate"],
                                 "Syn_recall": d["val"]["recall"].get("Syn", 0.0),
                                 "Syn_mean_p": d["val"]["mean_conf"].get("Syn", 0.0), "benign_far": d["val"]["benign_far"],
                                 "mean_w_flipped": tw["mean_w_flipped"], "mean_w_clean_known": tw["mean_w_clean_known"],
                                 "gate_reasons": "; ".join(d["gate_reasons"])})
            rs["arms"][arm] = ad
        rs["same_flips_in_both_poisoned_arms"] = all(v["poisoned"] == v["poisoned_repair"] for v in flips.values())
        rs["flips_persist"] = all(set(flips[t]["poisoned"]) <= set(flips[t + 1]["poisoned"]) for t in flips if t + 1 in flips)
        assert rs["same_flips_in_both_poisoned_arms"] and rs["flips_persist"], rdir.name
        summ["runs"][rdir.name] = rs
    df = pd.DataFrame(traj)
    df.to_csv(rcdir(a) / "trajectory_val.csv", index=False)
    pd.DataFrame(diag).to_csv(rcdir(a) / "diag_val.csv", index=False)
    summ["predictions_val"] = predictions(df, summ)
    summ["clean_arm_identical_across_rho"] = {
        str(seed): len({json.dumps(r["arms"]["clean"]["deployed"]) for r in summ["runs"].values() if r["seed"] == seed}) == 1
        for seed in a.seed_list}
    summ["timing"] = load_json(rcdir(a) / "timing.json") if (rcdir(a) / "timing.json").exists() else None
    summ["note"] = "validation only; no test was scored by any command that wrote this file"
    save_json(summ, rcdir(a) / "summary.json")
    pd.set_option("display.width", 220)
    piv = df.pivot_table(index=["rho", "seed", "arm"], columns="t", values="Syn_recall")
    print("deployed Syn val recall after each update t:\n", piv.round(3).to_string())
    print(json.dumps(summ["predictions_val"], indent=1))


def predictions(df, summ) -> dict:
    """The registered predictions RC1-RC4, as fixed in the prereg's clarifications (validation only)."""
    out = {}
    for rho in sorted(df["rho"].unique(), reverse=True):
        d = df[df["rho"] == rho]
        per = {}
        for seed, g in d.groupby("seed"):
            po, cl = g[g["arm"] == "poisoned"].set_index("t"), g[g["arm"] == "clean"].set_index("t")
            T = int(po.index.max())
            rc = summ["runs"][g["run"].iloc[0]]
            rep_dec = rc["arms"]["poisoned_repair"]["decisions"]
            first_rej = next((t for t in range(1, T + 1) if not d[(d["seed"] == seed) & (d["arm"] == "poisoned_repair")
                                                               & (d["t"] == t)]["c1_accepted"].iloc[0]), None)
            same_upto = [rc["arms"]["poisoned"]["deployed"][t]["state_sha256"] ==
                         rc["arms"]["poisoned_repair"]["deployed"][t]["state_sha256"]
                         for t in range(1, (first_rej or T + 1))]
            rho_s = pd.Series(po["Syn_mean_p"].to_numpy()).corr(pd.Series(po["flipped_share"].to_numpy()), method="spearman")
            per[str(seed)] = {
                "poisoned_C1_accepted": int(po.loc[1:, "c1_accepted"].sum()), "updates": T,
                "poisoned_drift": float(po.loc[0, "Syn_recall"] - po.loc[T, "Syn_recall"]),
                "poisoned_mean_p_A0": float(po.loc[0, "Syn_mean_p"]), "poisoned_mean_p_AT": float(po.loc[T, "Syn_mean_p"]),
                "poisoned_spearman_mean_p_vs_flips": float(rho_s),
                "repair_first_C1_rejection": first_rej, "repair_identical_to_poisoned_before_it": all(same_upto),
                "clean_accepts_all": bool(cl.loc[1:, "c1_accepted"].all()),
                "clean_max_abs_Syn_change": float((cl["Syn_recall"] - cl.loc[0, "Syn_recall"]).abs().max())}
        n = len(per)
        need = int(np.ceil(2 * n / 3))
        rc1a = sum(v["poisoned_C1_accepted"] >= v["updates"] - 1 for v in per.values())
        rc1b = sum(v["poisoned_drift"] > 0.05 for v in per.values())
        rc2 = sum(v["poisoned_spearman_mean_p_vs_flips"] < 0 and v["poisoned_mean_p_AT"] < v["poisoned_mean_p_A0"]
                  for v in per.values())
        rc4 = sum(v["clean_accepts_all"] and v["clean_max_abs_Syn_change"] <= 0.02 for v in per.values())
        out[f"rho_{rho:.2f}"] = {
            "per_seed": per,
            "RC1_ratchet": ({"C1_accepted_at_>=5_of_6_seeds": rc1a, "drift_>0.05_seeds": rc1b,
                             "met": rc1a >= need and rc1b >= need} if rho == 0.10 else "no prediction at rho 0.05"),
            "RC2_confidence_loss": ({"seeds": rc2, "met": rc2 == n} if rho == 0.10 else "registered at rho 0.10 only"),
            "RC3_repair_is_gated": {"runs_without_any_C1_rejection": sum(v["repair_first_C1_rejection"] is None
                                                                        for v in per.values()),
                                    "identical_models_before_first_rejection": sum(
                                        v["repair_identical_to_poisoned_before_it"] for v in per.values()),
                                    "met": all(v["repair_identical_to_poisoned_before_it"] for v in per.values())},
            "RC4_clean_control": {"seeds": rc4, "met": rc4 >= need}}
    return out


# ---- validation -> test handoff ------------------------------------------------
def val_files(a) -> list[Path]:
    """The validation decision files (small JSON/CSV; no buffers, no checkpoints). score-test checks each of them
    byte for byte against the decisions commit. rc/timing.json is zipped too but not checked: a restore session
    appends to it (its content at summarize time is inside summary.json, which is checked)."""
    w, (p, runs) = work(a), runs_done(a)
    rc = w / "rc"
    files = [rc / f for f in ("pilot.json", "summary.json", "trajectory_val.csv", "diag_val.csv")]
    files += sorted(rc.glob("schedule_*.json")) + ([rc / "stopped.json"] if (rc / "stopped.json").exists() else [])
    files += [w / x["tag"] / "cii" / "experience_manifest.json" for x in p["attempts"]]
    full = {r.name for _, _, r in runs}
    for x in p["attempts"]:  # pilot-only runs (clean arm only), e.g. a failed first pilot
        for s in a.seed_list:
            r = w / "runs" / "cii" / run_name(x["tag"], RHOS[0], s)
            if r.name not in full:
                files += [r / f for f in ("config.json", "run_manifest.json", "A0_metrics.json", "clean/rounds.json",
                                          "clean/val_done.json") if (r / f).exists()]
    for _, _, r in runs:
        files += [r / f for f in ("config.json", "run_manifest.json", "A0_metrics.json")]
        files += [r / arm / f for arm in ARMS for f in ("rounds.json", "val_done.json")] + [r / "poisoned" / "r1_diag.json"]
    return files


def deployed(a) -> list[Path]:
    _, runs = runs_done(a)
    return [r / "A0.pt" for _, _, r in runs] + [r / arm / f"A{t}.pt" for _, _, r in runs for arm in ARMS
                                                for t in range(1, len(load_json(r / "config.json")["schedule"]))]


def bundle(a):
    w = work(a)
    files = val_files(a) + [w / "rc" / "timing.json"] + deployed(a)
    manifest = {"files": {str(p.relative_to(w)): S3.file_hash(p) for p in files},
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": "RC validation outcome and deployed models only (no test). Restore with `s12_repeated.py restore`."}
    with zipfile.ZipFile(a.bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(w))
        zf.writestr("RESTORE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"{len(files)} files + RESTORE_MANIFEST.json -> {a.bundle} ({Path(a.bundle).stat().st_size / 1e6:.1f} MB)")


def zip_val(a):
    w = work(a)
    files = val_files(a) + [w / "rc" / "timing.json"]
    with zipfile.ZipFile(a.bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(w))
    print(f"{len(files)} validation files -> {a.bundle} ({Path(a.bundle).stat().st_size / 1e6:.2f} MB)")


def restore(a):
    src = Path(a.bundle)
    with tempfile.TemporaryDirectory() as tmp:
        if src.is_file():
            zipfile.ZipFile(src).extractall(tmp)
            src = Path(tmp)
        man = load_json(src / "RESTORE_MANIFEST.json")
        bad = {f: "missing" if not (src / f).exists() else "hash differs" for f, h in man["files"].items()
               if not (src / f).exists() or S3.file_hash(src / f) != h}
        assert not bad, f"refuse: bundle damaged {bad}"
        w = work(a)
        for f in man["files"]:
            t = w / f
            if t.exists():
                assert S3.file_hash(t) == man["files"][f], f"refuse: {t} exists with different content"
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, t)
    print(f"restored {len(man['files'])} files into {work(a)}; every sha256 matches the manifest")


def committed_copy(repo: Path, sha: str, rel: str) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{COMMITTED}/{rel}"], capture_output=True)
    assert p.returncode == 0, f"{COMMITTED}/{rel} not found in commit {sha} (fetch it first): {p.stderr.decode()[-300:]}"
    return p.stdout


def score_test(a):
    assert re.fullmatch(r"[0-9a-f]{7,40}", a.decisions_commit or ""), \
        "--decisions-commit <git sha> of the commit that holds the RC validation files is required"
    w = work(a)
    repo = Path(a.repo or HERE)
    for f in val_files(a):
        assert f.read_bytes() == committed_copy(repo, a.decisions_commit, str(f.relative_to(w))), \
            f"refuse: {f} differs from the copy committed in {a.decisions_commit}"
    tdir = w / "rc" / "test"
    if tdir.exists():
        raise SystemExit(f"refuse: {tdir} exists; RC test is scored once")
    summ = load_json(w / "rc" / "summary.json")
    ctx = S3.load_context(w, summ["tag"], "cii", a.device)[0]
    rows, hashes = [], set()
    for name, rs in summ["runs"].items():
        rdir = w / "runs" / "cii" / name
        cfg = load_json(rdir / "config.json")
        bad = [n for n, h in cfg["inputs"].items() if S3.file_hash(w / n) != h]
        assert not bad, f"refuse: Stage 1/1b files differ from the ones {name} used: {bad}"
        hp = Namespace(**cfg["args"])
        for arm in ARMS:
            out = []
            for t, d in enumerate(rs["arms"][arm]["deployed"]):
                path = rdir / "A0.pt" if t == 0 else rdir / arm / f"A{t}.pt"
                assert S3.file_hash(path) == d["file_sha256"], f"refuse: {path} is not the validated model"
                m, cls = S3.load_ckpt(path, ctx, hp)
                te = S3.evaluate(m, cls, ctx["test"], ctx["device"], scope=ctx["scope"])
                hashes.add(te["rows_sha256"])
                out.append({"t": t, "classes": cls, "test": te})
                rows.append({"run": name, "rho": rs["rho"], "seed": rs["seed"], "arm": arm, "t": t,
                             "Syn_recall": te["recall"].get("Syn", 0.0), "Syn_mean_p": te["mean_conf"].get("Syn", 0.0),
                             "benign_far": te["benign_far"], "n_benign": te["n_benign"],
                             "macro_recall": te["macro_recall"], "recall": json.dumps(te["recall"]),
                             "test_rows_sha256": te["rows_sha256"]})
            (tdir / name).mkdir(parents=True, exist_ok=True)
            save_json({"arm": arm, "decisions_commit": a.decisions_commit,
                       "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trajectory": out},
                      tdir / name / f"{arm}.json")
    assert len(hashes) == 1, "refuse: different test rows across models"
    df = pd.DataFrame(rows)
    df.to_csv(tdir / "test_trajectory.csv", index=False)
    print(df.pivot_table(index=["rho", "seed", "arm"], columns="t", values="Syn_recall").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "pilot", "run", "summarize", "zip-val", "bundle", "restore", "score-test"])
    ap.add_argument("--src", help="nids_out of notebook 04: clean_*.parquet and runs/cii/all12_cii_s42 (read only)")
    ap.add_argument("--work", default=None, help="writable nids_out for Stage 1b, runs and rc/ (default: work dir)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--attack-cap", type=int, default=ATTACK_CAP)
    ap.add_argument("--classes", default=",".join(REGISTERED_CLASSES), help="smoke test only; default = registered")
    ap.add_argument("--s1b-args", default="", help="smoke test only (synthetic data)")
    ap.add_argument("--s3-args", default="", help="smoke test only (synthetic data)")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None, help="git checkout holding the decisions commit (default: this code's)")
    ap.add_argument("--bundle", default=None, help="zip-val / bundle: zip to write; restore: zip or extracted folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    a.class_list = a.classes.split(",")
    {"prepare": prepare, "pilot": pilot, "run": run, "summarize": summarize, "zip-val": zip_val, "bundle": bundle,
     "restore": restore, "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
