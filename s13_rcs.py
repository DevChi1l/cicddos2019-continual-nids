"""RCS: repeated small replay corruption on a stability-screened schedule. A NEW experiment, separate from RC.

Pre-registration: experiments/prereg_rcs.md (committed before this code). RC (s12, notebook 08) stopped at its clean
pilot; its files are in experiments/results/08_rc/ and are not RCS results. RCS reuses s12's run / summarize /
bundle / restore / score-test code with its own layout: work folder rcs/, run names rcs_<tag>_r<rho>_s<seed>,
committed copies under experiments/results/09_rcs/.

  prepare     copy + hash-check the Stage 1 files (as s12) and build Stage 1b of both candidate tags; checks the
              registered protected val rows
  screen      Stage S: clean arm only, both candidates, seeds 42-44, validation only. Selects the first candidate
              (priority order) whose clean runs satisfy the stability predicate in 3/3 seeds -> rcs/screen.json
  confirm     Stage C: clean arm only, the selected candidate, fresh seeds 45-47 (inside the rho 0.10 run folders).
              Proceed only if the predicate holds in 3/3 seeds -> rcs/go.json (with the clean files' sha256)
  run         Stage P: refuses unless go.json says proceed. rho 0.10 (resumes the confirmation runs: their clean
              arm is the control, not retrained), then rho 0.05 unless the measured T4 time > 90 min
  summarize   checks the confirmation clean files are unchanged, then s12's trajectories and predictions
  zip-val / bundle / restore / score-test   as s12 (score-test checks experiments/results/09_rcs/)

Stability predicate (per clean run): (a) every C1 accepted; (b) every C1's smallest gate margin >= 0.01;
(c) max_t |Syn val recall(A_t) - Syn val recall(A0)| <= 0.02.

Usage (see notebooks/09_repeated_corruption_stable.ipynb):
  python s13_rcs.py prepare --src <nids_out of notebook 04> --work /kaggle/working/nids_out
"""
from __future__ import annotations

import argparse
import shlex
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch

import s12_repeated as S12
import s3_continual as S3
from common import load_json, save_json

# registered candidates, priority order (E0 = BENIGN + Syn); no others
CANDIDATES = {"rcs6": ["MSSQL", "NTP", "SNMP", "SSDP", "TFTP", "LDAP"],
              "rcs5": ["MSSQL", "NTP", "SNMP", "SSDP", "TFTP"]}
SCREEN_SEEDS, MAIN_SEEDS = (42, 43, 44), (45, 46, 47)
MIN_MARGIN, SYN_BAND = 0.01, 0.02
# protected val rows (Stage 1b on a local copy of the real rows, 2026-10-28, before registration; no training)
REGISTERED_VAL = {
    "rcs6": {"BENIGN": 9507, "Syn": 5973, "MSSQL": 5591, "NTP": 5994, "SNMP": 4955, "SSDP": 4719, "TFTP": 5993,
             "LDAP": 2301},
    "rcs5": {"BENIGN": 9507, "Syn": 5973, "MSSQL": 5591, "NTP": 5994, "SNMP": 4959, "SSDP": 4719, "TFTP": 5993}}
assert all(len(c) >= S12.MIN_UPDATES for c in CANDIDATES.values())
COMMITTED = "experiments/results/09_rcs"  # rcs_val.zip's contents (paths start with rcs/, runs/ or a tag)
CLEAN_FILES = ("config.json", "A0_metrics.json", "clean/rounds.json", "clean/val_done.json")


def run_name(tag, rho, seed):
    return f"rcs_{tag}_r{rho:.2f}_s{seed}"


def rcdir(a) -> Path:
    d = S12.work(a) / "rcs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidates(a) -> dict:
    """The registered candidates, or (smoke test only) --candidates 'tag=A,B;tag2=C'."""
    if not a.candidates:
        return dict(CANDIDATES)
    return {t: c.split(",") for t, c in (x.split("=") for x in a.candidates.split(";"))}


def go(a) -> dict | None:
    f = rcdir(a) / "go.json"
    return load_json(f) if f.exists() else None


def runs_done(a):
    """(go.json, finished main runs) in s12's format, so s12's summarize / bundle / score-test work unchanged."""
    g = go(a)
    out = []
    if g and g["proceed"]:
        for rho in S12.RHOS:
            for seed in a.seed_list:
                rdir = S12.work(a) / "runs" / "cii" / run_name(g["tag"], rho, seed)
                if S12.finished(rdir):
                    out.append((rho, seed, rdir))
    return g, out


def val_files(a) -> list[Path]:
    """The validation decision files (no buffers, no checkpoints): stage decisions, schedules, Stage 1b manifests,
    the clean files of every screening / confirmation run, and s12's per-run files of the finished main runs."""
    w, r = S12.work(a), rcdir(a)
    files = [r / f for f in ("screen.json", "go.json", "summary.json", "trajectory_val.csv", "diag_val.csv",
                             "stopped.json") if (r / f).exists()]
    files += sorted(r.glob("schedule_*.json"))
    files += [w / t / "cii" / "experience_manifest.json" for t in candidates(a)
              if (w / t / "cii" / "experience_manifest.json").exists()]
    full = {rd.name for _, _, rd in runs_done(a)[1]}
    for rd in sorted((w / "runs" / "cii").glob("rcs_*")):
        if rd.name in full:
            files += [rd / f for f in ("config.json", "run_manifest.json", "A0_metrics.json")]
            files += [rd / arm / f for arm in S12.ARMS for f in ("rounds.json", "val_done.json")]
            files += [rd / "poisoned" / "r1_diag.json"]
        else:
            files += [rd / f for f in ("config.json", "run_manifest.json", *CLEAN_FILES[1:]) if (rd / f).exists()]
    return files


def use_rcs_layout(a):
    """Point s12's shared code at the RCS layout (s12 looks these names up at call time)."""
    S12.rcdir, S12.run_name, S12.runs_done, S12.val_files, S12.COMMITTED = rcdir, run_name, runs_done, val_files, COMMITTED


# ---- prepare --------------------------------------------------------------------
def prepare(a):
    S12.copy_stage1(Path(a.src), S12.work(a))
    for tag, classes in candidates(a).items():
        m = S12.stage1b(a, tag, classes)
        if not a.candidates and a.attack_cap == S12.ATTACK_CAP:
            val = m["protected_exclusions"]["kept_by_split_class"]["val"]
            assert val == REGISTERED_VAL[tag], f"refuse: {tag} protected val rows {val} != registered {REGISTERED_VAL[tag]}"
            print(f"{tag}: protected val rows equal the registered feasibility check")


# ---- stages S and C -------------------------------------------------------------
def clean_runs(a, tag, seeds, label):
    """Clean arm only, inside the rho 0.10 run folder (validation only, no test)."""
    for seed in seeds:
        name = run_name(tag, S12.RHOS[0], seed)
        S12.call("s3_continual.py", "--out", S12.work(a), "--tag", tag, "--run", name, "--seed", seed,
                 *S12.s3_flags(a, S12.RHOS[0]), "--arms", "clean", "--resume",
                 timing=rcdir(a) / "timing.json", label=f"{label} {name}")


def stability(rdir: Path, n_updates: int, a) -> dict:
    """The registered predicate S for one clean run, from its saved validation files."""
    hp = Namespace(**load_json(rdir / "config.json")["args"])
    a0 = load_json(rdir / "A0_metrics.json")["val"]["recall"]["Syn"]
    recs = load_json(rdir / "clean" / "rounds.json")
    assert [r["attempt"] for r in recs] == ["C1"] * n_updates, f"{rdir.name}: expected {n_updates} C1 attempts"
    ups = []
    for r in recs:
        m, k = S12.margins(r, hp)
        deployed = r["val"] if r["accepted"] else r["ref_val"]  # rejected -> the previous model stays
        ups.append({"t": r["experience"], "new_class": S12.new_label(r), "accepted": bool(r["accepted"]),
                    "min_margin": m[k], "binding": k, "Syn_recall": deployed["recall"]["Syn"],
                    "gate_reasons": r["gate_reasons"]})
    acc = all(u["accepted"] for u in ups)
    margin = min(u["min_margin"] for u in ups)
    syn = max(abs(u["Syn_recall"] - a0) for u in ups)
    return {"run": rdir.name, "A0_Syn_recall": a0, "updates": ups, "all_accepted": acc, "min_margin": margin,
            "max_abs_Syn_change": syn, "stable": acc and margin >= a.min_margin and syn <= a.syn_band}


def rule_text(a):
    return (f"stable run: every C1 accepted, every C1's smallest gate margin >= {a.min_margin}, "
            f"max |Syn val recall(A_t) - A0| <= {a.syn_band}; a stage passes if every seed is stable")


def screen(a):
    out = rcdir(a) / "screen.json"
    if out.exists():
        s = load_json(out)
        print(f"screen decided earlier: selected={s['selected']}")
        return
    cands, res = candidates(a), {}
    for tag, classes in cands.items():  # both candidates always run; selection by priority order
        S12.stage1b(a, tag, classes)
        clean_runs(a, tag, a.screen_seed_list, "screen")
        per = {str(s): stability(S12.work(a) / "runs" / "cii" / run_name(tag, S12.RHOS[0], s), len(classes), a)
               for s in a.screen_seed_list}
        res[tag] = {"classes": classes, "stable_seeds": sum(v["stable"] for v in per.values()), "per_seed": per}
    selected = next((t for t in cands if res[t]["stable_seeds"] == len(a.screen_seed_list)), None)
    save_json({"selected": selected, "classes": cands[selected] if selected else None,
               "priority": list(cands), "seeds": list(a.screen_seed_list), "candidates": res,
               "rule": rule_text(a) + "; select the first candidate in priority order that passes; none -> STOP",
               "min_margin": a.min_margin, "syn_band": a.syn_band,
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, out)
    for t, r in res.items():
        print(f"{t}: stable in {r['stable_seeds']}/{len(a.screen_seed_list)} seeds")
    print(f"screen: selected={selected}" + ("" if selected else " -> STOP (no candidate is stable)"))


def confirm(a):
    out = rcdir(a) / "go.json"
    if out.exists():
        g = load_json(out)
        print(f"confirmation decided earlier: proceed={g['proceed']} tag={g['tag']} ({g['reason']})")
        return
    if not (rcdir(a) / "screen.json").exists():
        raise SystemExit("refuse: run the screen (stage S) first")
    s = load_json(rcdir(a) / "screen.json")
    if not s["selected"]:
        raise SystemExit("refuse: the screen selected no candidate (STOP)")
    tag, classes = s["selected"], s["classes"]
    S12.stage1b(a, tag, classes)
    clean_runs(a, tag, a.seed_list, "confirm")
    w = S12.work(a)
    per, hashes = {}, {}
    for seed in a.seed_list:
        rdir = w / "runs" / "cii" / run_name(tag, S12.RHOS[0], seed)
        per[str(seed)] = stability(rdir, len(classes), a)
        hashes.update({str((rdir / f).relative_to(w)): S3.file_hash(rdir / f) for f in CLEAN_FILES})
    n = sum(v["stable"] for v in per.values())
    proceed = n == len(a.seed_list)
    save_json({"proceed": proceed, "tag": tag, "classes": classes, "seeds": list(a.seed_list),
               "reason": f"stable in {n}/{len(a.seed_list)} confirmation seeds", "per_seed": per,
               "clean_files_sha256": hashes, "rule": rule_text(a) + "; proceed to poisoned runs only if it passes",
               "min_margin": a.min_margin, "syn_band": a.syn_band,
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, out)
    print(f"confirmation: PROCEED = {proceed} ({n}/{len(a.seed_list)} seeds stable, tag {tag})")


# ---- stage P --------------------------------------------------------------------
def run(a):
    g = go(a)
    if not g or not g["proceed"]:
        raise SystemExit("refuse: no go.json with proceed = true (stages S and C decide; do not change the design)")
    timing = rcdir(a) / "timing.json"
    for i, rho in enumerate(S12.RHOS):
        total = load_json(timing)["total_seconds"]
        if i > 0 and total > S12.T4_CAP_S:
            save_json({"stopped_before_rho": rho, "total_seconds": total, "rule": "stopping rule 3: 90 min of T4"},
                      rcdir(a) / "stopped.json")
            print(f"stopping rule 3: {total / 60:.0f} min > 90 min after rho {S12.RHOS[i - 1]}; rho {rho} not run")
            return
        for seed in a.seed_list:
            name = run_name(g["tag"], rho, seed)
            if S12.finished(S12.work(a) / "runs" / "cii" / name):
                print(f"{name}: finished earlier, skipped")
                continue
            S12.call("s3_continual.py", "--out", S12.work(a), "--tag", g["tag"], "--run", name, "--seed", seed,
                     *S12.s3_flags(a, rho), "--arms", ",".join(S12.ARMS), "--resume", timing=timing,
                     label=f"run {name}")
    print(f"all runs done; measured time {load_json(timing)['total_seconds'] / 60:.1f} min")


def summarize(a):
    g = go(a)
    w = S12.work(a)
    bad = [f for f, h in g["clean_files_sha256"].items() if S3.file_hash(w / f) != h]
    assert not bad, f"refuse: confirmation clean files changed after go.json: {bad}"
    S12.summarize(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "screen", "confirm", "run", "summarize", "zip-val", "bundle", "restore",
                                    "score-test"])
    ap.add_argument("--src", help="nids_out of notebook 04: clean_*.parquet and runs/cii/all12_cii_s42 (read only)")
    ap.add_argument("--work", default=None, help="writable nids_out for Stage 1b, runs and rcs/ (default: work dir)")
    ap.add_argument("--attack-cap", type=int, default=S12.ATTACK_CAP)
    ap.add_argument("--candidates", default="", help="smoke test only: 'tag=A,B;tag2=C' (default = registered)")
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN, help="smoke test only")
    ap.add_argument("--syn-band", type=float, default=SYN_BAND, help="smoke test only")
    ap.add_argument("--s1b-args", default="", help="smoke test only (synthetic data)")
    ap.add_argument("--s3-args", default="", help="smoke test only (synthetic data)")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None, help="git checkout holding the decisions commit (default: this code's)")
    ap.add_argument("--bundle", default=None, help="zip-val / bundle: zip to write; restore: zip or extracted folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.seed_list, a.screen_seed_list = list(MAIN_SEEDS), list(SCREEN_SEEDS)
    use_rcs_layout(a)
    {"prepare": prepare, "screen": screen, "confirm": confirm, "run": run, "summarize": summarize,
     "zip-val": S12.zip_val, "bundle": S12.bundle, "restore": S12.restore, "score-test": S12.score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
