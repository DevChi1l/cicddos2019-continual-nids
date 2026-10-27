"""Follow-up FU2 - deploy teacher relabelling as the repair at 100% Syn replay poisoning (four-class CII).

Pre-registration: experiments/prereg_followup_fu2.md (committed before any FU2 run).

In the sweep (notebook 04) at f = 1.0, the registered soft R1 failed every time: all stored Syn labels were
BENIGN, so down-weighting them left no Syn label. "threeway+relabel" was only a forced DIAGNOSTIC there (never
a decision, never deployed, never tested); it would have passed the gate 6 of 6 times. FU2 deploys it.

Per seed, FU2 runs ONE new arm, `poisoned_relabel`, with s3's own code (S3.run_arm): the source run's frozen T
(read only, hash-checked), its A0 (copied, hash-checked), the same poisoned buffer and seeds as its
poisoned_repair arm. Only R1's method changes, soft -> threeway+relabel:
  p_T(stored label) >= 0.9 -> weight 1; <= 0.1 -> weight 0; otherwise 0.5; and rows with p_T(stored) <= 0.1 and
  max_k p_T(k) >= 0.9 get T's predicted label (weight 1). Rows of classes T never saw keep label and weight 1.
Flips persist: relabelling changes only the labels R1 trains on, never the stored buffer. Gate, splits and
hyperparameters are the source run's. The clean / poisoned / poisoned_repair arms are the controls: read from the
source run, never rerun.

  run         validation only, all seeds; finished seeds are skipped, an unfinished one restarts (deterministic)
  bundle      deployed models + summaries + rounds + sha256 manifest in one zip (no test)
  restore     a later session: verify every hash in a bundle, then copy it into <work>/fu2/
  score-test  each seed's deployed model on test, ONCE. Refuses unless summary.json and rounds.json are
              byte-identical to their copies under experiments/results/06_fu2/ in the given git commit, and the
              model has the hash recorded in summary.json

Usage (see notebooks/06_followup_fu2_relabel.ipynb):
  python s10_relabel.py run --src <nids_out of notebook 04> --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import s3_continual as S3
from common import default_work_dir, load_json, save_json

ARM = "poisoned_relabel"
METHOD = {"repair_rule": "threeway", "relabel_to_teacher": True, "tw_keep": 0.9, "tw_quarantine": 0.1,
          "tw_mid_weight": 0.5}
COMMITTED = "experiments/results/06_fu2"  # where fu2_val.zip's contents are committed (paths start with fu2/)
BUNDLED = ("summary.json", f"{ARM}/rounds.json")  # per seed, plus the deployed model named in summary.json


def source(src: Path, seed: int, frac: float):
    run = src / "runs" / "cii" / f"sweep_f{frac}_s{seed}"
    cfg = load_json(run / "config.json")
    hp = Namespace(**cfg["args"])
    assert hp.poison_frac == frac and hp.seed == seed and not hp.tag, f"{run} is not the f={frac} seed {seed} run"
    assert all(getattr(hp, k) == v for k, v in METHOD.items() if k.startswith("tw_")), "tw thresholds differ"
    bad = [n for n in S3.input_names("", "cii") if S3.file_hash(src / n) != cfg["inputs"][n]]
    assert not bad, f"Stage 1/1b files differ from the ones {run.name} used: {bad}"
    return run, cfg, hp


def fu2_dir(a) -> Path:
    return Path(a.work or default_work_dir()) / "fu2"


def run_dir(a, seed: int) -> Path:
    return fu2_dir(a) / "cii" / f"fu2_relabel_f{a.frac}_s{seed}"


def tree_hash(d: Path) -> dict:
    return {str(p.relative_to(d)): S3.file_hash(p) for p in sorted(d.rglob("*")) if p.is_file()}


def same_model(p1: Path, p2: Path, ctx, hp) -> bool | None:
    if not (p1.exists() and p2.exists()):
        return None
    return S3.state_hash(S3.load_ckpt(p1, ctx, hp)[0]) == S3.state_hash(S3.load_ckpt(p2, ctx, hp)[0])


def run(a):
    src, ctx = Path(a.src), None
    for seed in map(int, a.seeds.split(",")):
        srun, cfg, hp = source(src, seed, a.frac)
        rdir = run_dir(a, seed)
        assert not rdir.resolve().is_relative_to((src / "runs").resolve()), "never write into the s3 runs"
        if (rdir / "summary.json").exists():
            print(f"s{seed}: finished earlier, skipped (never overwritten)")
            continue
        if ctx is None:
            ctx = S3.load_context(src, "", "cii", a.device)[0]
        man = load_json(srun / "run_manifest.json")
        before = tree_hash(srun)
        T, T_cls = S3.load_ckpt(srun / "T.pt", ctx, hp)
        T.eval()
        for p in T.parameters():
            p.requires_grad_(False)
        assert S3.state_hash(T) == man["T_sha256"], "not the source run's frozen teacher"
        if a.dry_run:  # rebuild the poisoned buffer as s3 does; it must equal the source run's saved one
            buf = S3.buffer_add(pd.DataFrame(columns=["rid", "label", "true_label", "flipped"]), ctx["train"],
                                ctx["exp"][0], hp.mem_per_class, np.random.default_rng(seed + 1))
            buf = S3.poison(buf, hp.poison_target, hp.poison_to, hp.poison_frac, np.random.default_rng(seed + 2))
            same = buf.to_csv(index=False) == (srun / "poisoned_repair" / "buffer_before_E1.csv").read_text()
            print(f"s{seed}: inputs match, T matches run_manifest, rebuilt poisoned buffer equals the saved one: "
                  f"{same} ({int(buf['flipped'].sum())} flipped)")
            assert same
            continue
        if rdir.exists():
            print(f"s{seed}: unfinished earlier attempt removed; restarting this seed (deterministic)")
            shutil.rmtree(rdir)
        rdir.mkdir(parents=True)
        shutil.copy2(srun / "A0.pt", rdir / "A0.pt")
        assert S3.file_hash(rdir / "A0.pt") == S3.file_hash(srun / "A0.pt")
        hp_r = Namespace(**{**vars(hp), **METHOD})
        print(f"--- seed {seed}: arm {ARM}, R1 = threeway+relabel, source {srun.name} ---", flush=True)
        records, _ = S3.run_arm(ARM, ctx, hp_r, rdir, T, T_cls, score_test=False)
        assert S3.state_hash(T) == man["T_sha256"] and tree_hash(srun) == before, "teacher or source run changed"

        last = len(ctx["schedule"]) - 1
        final = rdir / ARM / f"A{last}.pt"
        _, final_cls = S3.load_ckpt(final, ctx, hp)
        adir = rdir / ARM
        checks = {  # P0: FU2's E1 must replay the sweep exactly (same start, buffer, seeds, method)
            "buffer_E1_equals_source": (adir / "buffer_before_E1.csv").read_bytes()
                                       == (srun / "poisoned_repair" / "buffer_before_E1.csv").read_bytes(),
            "E1_C1_equals_source_C1": same_model(adir / "E1_C1.pt", srun / "poisoned_repair" / "E1_C1.pt", ctx, hp),
            "E1_R1_equals_forced_diagnostic": same_model(adir / "E1_R1.pt",
                                                         srun / "poisoned" / "E1_R1diag_threeway_relabel.pt", ctx, hp),
        }
        status = {}
        for t in range(1, last + 1):
            att = [r for r in records if r["experience"] == t]
            acc = [r["attempt"] for r in att if r["accepted"]]
            status[f"E{t}"] = f"accepted at {acc[0]}" if acc else "rejected (C1, R1): previous model kept"
        relabel = [{"experience": r["experience"], **{k: r["teacher_weights"][k] for k in
                    ("n_relabelled", "n_relabelled_to_true", "n_relabelled_flipped", "n_quarantined",
                     "n_quarantined_flipped", "mean_w_flipped", "mean_w_clean_known")}}
                   for r in records if r["attempt"] == "R1"]
        save_json({"source_run": srun.name, "seed": seed, "poison_frac": a.frac, "arm": ARM,
                   "r1_method": "threeway+relabel", "method_args": METHOD, "inputs": cfg["inputs"],
                   "T_sha256": man["T_sha256"], "T_unchanged": True, "source_run_unchanged": True,
                   "A0_sha256": S3.file_hash(rdir / "A0.pt"), "final_ckpt": f"{ARM}/A{last}.pt",
                   "final_sha256": S3.file_hash(final), "final_classes": final_cls, "update_status": status,
                   "relabel_stats": relabel, "p0_checks": checks, "device": str(ctx["device"]),
                   "torch": torch.__version__}, rdir / "summary.json")
        print(f"s{seed}: {status} | deployed classes {final_cls} | P0 {checks}", flush=True)


def bundle(a):
    """Restorable copy of the validation outcome: no test file can enter it."""
    root = fu2_dir(a)
    runs = sorted(p.parent for p in root.glob("cii/*/summary.json"))
    assert runs, f"no finished FU2 run under {root}"
    files = [r / f for r in runs for f in BUNDLED + (load_json(r / "summary.json")["final_ckpt"],)]
    for r in runs:
        s = load_json(r / "summary.json")
        assert s["final_sha256"] == S3.file_hash(r / s["final_ckpt"]), f"{r}: deployed model differs from summary"
    manifest = {"files": {str(Path("fu2") / p.relative_to(root)): S3.file_hash(p) for p in files},
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": "FU2 validation outcome only (no test). Restore with `s10_relabel.py restore`."}
    with zipfile.ZipFile(a.bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, Path("fu2") / p.relative_to(root))
        zf.writestr("RESTORE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"{len(files)} files + RESTORE_MANIFEST.json -> {a.bundle} ({Path(a.bundle).stat().st_size / 1e6:.2f} MB)")


def restore(a):
    """Copy a bundle (zip, or the folder Kaggle extracts it to) into <work>/fu2 after checking every hash."""
    src = Path(a.bundle)
    with tempfile.TemporaryDirectory() as tmp:
        if src.is_file():
            zipfile.ZipFile(src).extractall(tmp)
            src = Path(tmp)
        man = load_json(src / "RESTORE_MANIFEST.json")
        bad = {f: "missing" if not (src / f).exists() else "hash differs" for f, h in man["files"].items()
               if not (src / f).exists() or S3.file_hash(src / f) != h}
        assert not bad, f"refuse: bundle damaged {bad}"
        for s in (src / "fu2").glob("cii/*/summary.json"):
            j = load_json(s)
            assert j["final_sha256"] == man["files"][str(s.parent.relative_to(src) / j["final_ckpt"])]
        root = Path(a.work or default_work_dir())
        for f in man["files"]:
            t = root / f
            if t.exists():
                assert S3.file_hash(t) == man["files"][f], f"refuse: {t} exists with different content"
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, t)
    print(f"restored {len(man['files'])} files into {root / 'fu2'}; every sha256 matches the manifest")


def committed_copy(repo: Path, sha: str, rel: str) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{COMMITTED}/{rel}"], capture_output=True)
    assert p.returncode == 0, f"{COMMITTED}/{rel} not found in commit {sha} (fetch it first): {p.stderr.decode()[-300:]}"
    return p.stdout


def score_test(a):
    assert re.fullmatch(r"[0-9a-f]{7,40}", a.decisions_commit or ""), \
        "--decisions-commit <git sha> of the commit that holds the FU2 validation files is required"
    src, root = Path(a.src), fu2_dir(a)
    repo = Path(a.repo or Path(__file__).resolve().parent)
    seeds = list(map(int, a.seeds.split(",")))
    for seed in seeds:  # every check passes for every seed before anything is scored
        rdir = run_dir(a, seed)
        assert (rdir / "summary.json").exists(), f"s{seed}: run `run` (or `restore`) first"
        if (rdir / "final_test.json").exists():
            raise SystemExit(f"refuse: {rdir} was already scored on test; it is scored once")
        for f in BUNDLED:
            assert (rdir / f).read_bytes() == committed_copy(repo, a.decisions_commit,
                                                             str((rdir / f).relative_to(root.parent))), \
                f"refuse: {rdir / f} differs from the copy committed in {a.decisions_commit}"
        s = load_json(rdir / "summary.json")
        assert S3.file_hash(rdir / s["final_ckpt"]) == s["final_sha256"], "refuse: not the validated model"
        assert s["inputs"] == source(src, seed, a.frac)[1]["inputs"]
    ctx, rows = S3.load_context(src, "", "cii", a.device)[0], []
    for seed in seeds:
        rdir, (srun, _, hp) = run_dir(a, seed), source(src, seed, a.frac)
        s = load_json(rdir / "summary.json")
        m, cls = S3.load_ckpt(rdir / s["final_ckpt"], ctx, hp)
        test = S3.evaluate(m, cls, ctx["test"], ctx["device"], scope=ctx["scope"])
        save_json({"arm": ARM, "final_classes": cls, "decisions_commit": a.decisions_commit,
                   "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test": test},
                  rdir / "final_test.json")
        for arm in ("clean", "poisoned", "poisoned_repair"):  # controls: scored by the sweep, same rows
            old = load_json(srun / arm / "final_test.json")["test"]
            assert old["rows_sha256"] == test["rows_sha256"], "not the source run's test rows"
            rows.append({"seed": seed, "arm": arm, **old["recall"], "benign_far": old["benign_far"]})
        rows.append({"seed": seed, "arm": ARM, **test["recall"], "benign_far": test["benign_far"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "bundle", "restore", "score-test"])
    ap.add_argument("--src", help="nids_out of notebook 04: clean_*.parquet, cii/ and runs/cii/sweep_f1.0_s* (read only)")
    ap.add_argument("--work", default=None, help="writes go to <work>/fu2/ (default: the nids_out work dir)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--frac", type=float, default=1.0, help="poison fraction of the source sweep runs (registered: 1.0)")
    ap.add_argument("--dry-run", action="store_true", help="run: check inputs, T and the poisoned buffer; no training")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None, help="git checkout holding the decisions commit (default: this code's)")
    ap.add_argument("--bundle", default=None, help="bundle: zip to write; restore: zip or extracted folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"run": run, "bundle": bundle, "restore": restore, "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
