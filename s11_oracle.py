"""Joint-training oracle (diagnostic): is the 12-label failure continual forgetting, or do the features fail?

Pre-registration: experiments/prereg_oracle.md (committed before any oracle run).

One model is trained on ALL training rows of the capped 12-label runs at once (the union of a scenario's
experiences: every class present from the start, no updates, no gate, no buffer). Same scaler, architecture,
loss and E0 recipe as the continual runs; scored on the same protected val rows (and, once, test rows). If
joint training cannot reach the gate's 0.80 new-class bar for DNS/LDAP or SSDP/UDP, the continual protocol is
not what fails them. If it can, the continual protocol is (forgetting / interference under the gate).

  train       registered settings x seeds x scenarios -> <work>/oracle/<sc>/<setting>_s<seed>/{model.pt, val.json};
              finished cells are skipped (resumable); validation only
  decide      the pre-registered verdict rule -> oracle/oracle_val.csv, oracle/verdict.json (validation only)
  bundle      models + val files + sha256 manifest (no test), for a later session
  restore     verify every hash in a bundle, then copy it into <work>/oracle/
  score-test  every model on test ONCE; refuses unless the val files are byte-identical to their copies under
              experiments/results/07_oracle/ in the given git commit and each model has its recorded hash

Usage (see notebooks/07_oracle_joint.ipynb):
  python s11_oracle.py train --src <nids_out of notebook 04>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
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

TAG = "all12"
SETTINGS = {"e10": 10, "e20": 20}  # e10 = the continual runs' E0 recipe; e20 = the one registered sensitivity check
CELLS = [("cii", s, seed) for s in SETTINGS for seed in (42, 43, 44)] + [("ci", s, 42) for s in SETTINGS]
PAIRS = [("DNS", "LDAP"), ("SSDP", "UDP")]
COMMITTED = "experiments/results/07_oracle"  # oracle_val.zip's contents (paths start with oracle/)
RULE = ("primary: scenario cii, median over seeds 42-44 of oracle VAL recall; per pair, the setting (e10/e20) whose "
        "median recall of the pair's weaker class is higher. Pair verdict: both classes >= 0.80 and median benign FAR "
        "<= 0.05 -> 'separable for the gate' (the 12-label failure of this pair is primarily continual); weaker class "
        "< 0.70 -> 'feature limit' (joint training also fails); otherwise 'borderline'. Overall: both pairs separable "
        "-> 'primarily continual'; both feature limit -> 'primarily features'; else 'mixed'. Any other class with "
        "oracle median val recall >= 0.80 that a continual model lost (test recall 0) is listed as lost by the protocol.")


def source(src: Path, sc: str):
    run = src / "runs" / sc / f"all12_{sc}_s42"
    cfg = load_json(run / "config.json")
    bad = [n for n in S3.input_names(TAG, sc) if S3.file_hash(src / n) != cfg["inputs"][n]]
    assert not bad, f"Stage 1/1b files differ from the ones {run.name} used: {bad}"
    return run, cfg, Namespace(**cfg["args"])


def odir(a) -> Path:
    return Path(a.work or default_work_dir()) / "oracle"


def train(a):
    src, ctxs = Path(a.src), {}
    for sc, setting, seed in CELLS:
        d = odir(a) / sc / f"{setting}_s{seed}"
        assert not d.resolve().is_relative_to((src / "runs").resolve()), "never write into the s3 runs"
        if (d / "val.json").exists():
            print(f"{sc} {setting} s{seed}: finished earlier, skipped")
            continue
        run, cfg, hp = source(src, sc)
        if sc not in ctxs:
            ctxs.clear()  # one scenario's rows in memory at a time
            ctxs[sc] = S3.load_context(src, TAG, sc, a.device)[0]
        ctx = ctxs[sc]
        tr = ctx["train"]
        i = tr.idx(np.sort(np.concatenate(ctx["exp"])))
        classes = ["BENIGN"] + [k for e in ctx["schedule"] for k in e]  # the continual runs' head order
        assert sorted(set(tr.lab[i])) == sorted(classes)
        torch.manual_seed(seed)  # as s3's E0: init, then batch order from `seed`
        m = S3.build(len(classes), ctx, hp)
        t0 = time.time()
        losses = S3.train(m, classes, tr.x[i], tr.c[i], tr.lab[i], np.ones(len(i)), SETTINGS[setting], hp,
                          ctx["device"], seed)
        secs = time.time() - t0
        val = S3.evaluate(m, classes, ctx["val"], ctx["device"])
        assert val["support"] == cfg["val_rows"], "not the continual runs' protected val rows"
        d.mkdir(parents=True, exist_ok=True)
        S3.save_ckpt(d / "model.pt", m, classes, {"role": "joint-training oracle", "setting": setting})
        save_json({"scenario": sc, "setting": setting, "epochs": SETTINGS[setting], "seed": seed, "classes": classes,
                   "train_rows": int(len(i)), "train_rows_sha256": S3.rid_hash(tr.rid[i]),
                   "train_rows_by_class": {k: int(v) for k, v in pd.Series(tr.lab[i]).value_counts().items()}, "inputs": cfg["inputs"],
                   "source_run": run.name, "hp": vars(hp), "losses": losses, "seconds": secs,
                   "model_sha256": S3.file_hash(d / "model.pt"), "device": str(ctx["device"]),
                   "torch": torch.__version__, "val": val}, d / "val.json")
        print(f"{sc} {setting} s{seed}: {secs:.0f}s | val recall " +
              " ".join(f"{k} {v:.3f}" for k, v in val["recall"].items()) + f" | benign FAR {val['benign_far']:.4f}",
              flush=True)


def decide(a):
    root = odir(a)
    rows = []
    for sc, setting, seed in CELLS:
        v = load_json(root / sc / f"{setting}_s{seed}" / "val.json")["val"]
        rows.append({"scenario": sc, "setting": setting, "seed": seed, "benign_far": v["benign_far"],
                     **{f"recall_{k}": r for k, r in v["recall"].items()},
                     **{f"{x}_to_{y}": v["confusion"][x].get(y, 0) / v["support"][x] for p in PAIRS
                        for x, y in (p, p[::-1]) if x in v["support"] and y in v["support"]}})
    df = pd.DataFrame(rows)
    df.to_csv(root / "oracle_val.csv", index=False)
    med = df[df["scenario"] == "cii"].groupby("setting").median(numeric_only=True)
    pairs = {}
    for x, y in PAIRS:
        if f"recall_{x}" not in med or f"recall_{y}" not in med:
            pairs[f"{x}/{y}"] = {"verdict": "not in this schedule"}  # synthetic smoke data only
            continue
        weak = med[[f"recall_{x}", f"recall_{y}"]].min(axis=1)
        s = weak.idxmax()
        rx, ry, far = med.loc[s, f"recall_{x}"], med.loc[s, f"recall_{y}"], med.loc[s, "benign_far"]
        v = ("separable for the gate" if min(rx, ry) >= 0.80 and far <= 0.05 else
             "feature limit" if min(rx, ry) < 0.70 else "borderline")
        pairs[f"{x}/{y}"] = {"setting": s, f"median_val_recall_{x}": rx, f"median_val_recall_{y}": ry,
                             "median_benign_far": far, "verdict": v}
    vs = [p["verdict"] for p in pairs.values()]
    overall = ("primarily continual" if all(v == "separable for the gate" for v in vs) else
               "primarily features" if all(v == "feature limit" for v in vs) else "mixed")
    best = med.max()  # per class, the better setting (descriptive)
    others = {k[len("recall_"):]: float(best[k]) for k in med.columns if k.startswith("recall_")
              and k[len("recall_"):] not in {c for p in PAIRS for c in p}}
    out = {"rule": RULE, "pairs": pairs, "overall": overall, "other_classes_best_median_val_recall": others,
           "note": "validation only; 'lost by the protocol' is filled in from committed continual test tables in the "
                   "results README, not by this command"}
    save_json(out, root / "verdict.json")
    print(df.drop(columns=[c for c in df.columns if "_to_" in c]).round(3).to_string(index=False))
    print(json.dumps({"pairs": pairs, "overall": overall}, indent=1, default=float))


def run_dirs(root: Path) -> list[Path]:
    return [root / sc / f"{s}_s{seed}" for sc, s, seed in CELLS]


def bundle(a):
    root = odir(a)
    files = [root / "oracle_val.csv", root / "verdict.json"] + [d / f for d in run_dirs(root)
                                                                for f in ("val.json", "model.pt")]
    for d in run_dirs(root):
        assert load_json(d / "val.json")["model_sha256"] == S3.file_hash(d / "model.pt"), f"{d}: model changed"
    manifest = {"files": {str(Path("oracle") / p.relative_to(root)): S3.file_hash(p) for p in files},
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": "oracle validation outcome only (no test). Restore with `s11_oracle.py restore`."}
    with zipfile.ZipFile(a.bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, Path("oracle") / p.relative_to(root))
        zf.writestr("RESTORE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"{len(files)} files + RESTORE_MANIFEST.json -> {a.bundle} ({Path(a.bundle).stat().st_size / 1e6:.2f} MB)")


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
        root = Path(a.work or default_work_dir())
        for f in man["files"]:
            t = root / f
            if t.exists():
                assert S3.file_hash(t) == man["files"][f], f"refuse: {t} exists with different content"
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, t)
    print(f"restored {len(man['files'])} files into {root / 'oracle'}; every sha256 matches the manifest")


def committed_copy(repo: Path, sha: str, rel: str) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{COMMITTED}/{rel}"], capture_output=True)
    assert p.returncode == 0, f"{COMMITTED}/{rel} not found in commit {sha} (fetch it first): {p.stderr.decode()[-300:]}"
    return p.stdout


def score_test(a):
    assert re.fullmatch(r"[0-9a-f]{7,40}", a.decisions_commit or ""), \
        "--decisions-commit <git sha> of the commit that holds the oracle validation files is required"
    src, root = Path(a.src), odir(a)
    repo = Path(a.repo or Path(__file__).resolve().parent)
    for p in [root / "oracle_val.csv", root / "verdict.json"] + [d / "val.json" for d in run_dirs(root)]:
        assert p.read_bytes() == committed_copy(repo, a.decisions_commit, str(p.relative_to(root.parent))), \
            f"refuse: {p} differs from the copy committed in {a.decisions_commit}"
    for d in run_dirs(root):
        if (d / "test.json").exists():
            raise SystemExit(f"refuse: {d} was already scored on test; it is scored once")
        assert S3.file_hash(d / "model.pt") == load_json(d / "val.json")["model_sha256"], f"refuse: {d} model changed"
    rows, ctx_sc, ctx = [], None, None
    for d, (sc, setting, seed) in zip(run_dirs(root), CELLS):
        run, cfg, hp = source(src, sc)
        if ctx_sc != sc:
            ctx, ctx_sc = None, sc
            ctx = S3.load_context(src, TAG, sc, a.device)[0]
        m, cls = S3.load_ckpt(d / "model.pt", ctx, hp)
        test = S3.evaluate(m, cls, ctx["test"], ctx["device"], scope=ctx["scope"])
        assert test["rows_sha256"] == load_json(run / "clean" / "final_test.json")["test"]["rows_sha256"], \
            "not the continual runs' test rows"
        save_json({"decisions_commit": a.decisions_commit,
                   "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test": test}, d / "test.json")
        rows.append({"scenario": sc, "setting": setting, "seed": seed, **test["recall"], "benign_far": test["benign_far"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "decide", "bundle", "restore", "score-test"])
    ap.add_argument("--src", help="nids_out of notebook 04: clean_*.parquet, all12/ and runs/{cii,ci}/all12_*_s42 (read only)")
    ap.add_argument("--work", default=None, help="writes go to <work>/oracle/ (default: the nids_out work dir)")
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--repo", default=None, help="git checkout holding the decisions commit (default: this code's)")
    ap.add_argument("--bundle", default=None, help="bundle: zip to write; restore: zip or extracted folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"train": train, "decide": decide, "bundle": bundle, "restore": restore, "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
