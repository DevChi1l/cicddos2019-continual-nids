"""Export everything the offline dashboard needs into ONE json file (no training unless a model is missing).

What it exports (f = 1.0, seed 42, four-class CII; the setting of the primary result):
  models   the deployed A2 of four arms: clean, poisoned (gate only), poisoned_repair (soft R1, registered),
           poisoned_relabel (FU2 relabel R1, post hoc). Raw float32 weights, so the browser can run the model.
  scaler   the frozen E0 scaler (signed log -> z-score -> clip) and the protocol vocabulary.
  rows     300 held-out TEST rows per class (seeded sample), RAW feature values, so the dashboard applies the
           scaler itself and a user can edit raw values.
  ref      PyTorch probabilities of every model on every exported row (CPU, eval mode). The dashboard re-computes
           them in JavaScript on load and shows the match count: proof that the page runs the real models.
  full     each model's metrics on the FULL test set (46,527 rows), asserted equal to the committed numbers
           (experiments/results/04_sweep/final_test.csv and 06_fu2/test/...), so the exported models ARE the
           reported models.

Where models come from (first match wins):
  1. notebook 04 output:  <src>/runs/cii/sweep_f1.0_s42/{clean,poisoned,poisoned_repair}/A2.pt
  2. FU2:  .../fu2/cii/fu2_relabel_f1.0_s42/poisoned_relabel/A2.pt, or inside a fu2_restore.zip
  3. --rebuild: re-run the missing pieces with the committed code (s3 sweep_f1.0_s42, then s10 seed 42); runs are
     deterministic, and the full-test check below catches any difference.

Usage (Kaggle, GPU optional, Internet ON):
  python make_dashboard_assets.py --out /kaggle/working/dashboard_assets.json [--rebuild]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import s3_continual as S3
import s5_explain as X
from common import LABEL_COL, load_json

HERE = Path(__file__).resolve().parent
RUN = "sweep_f1.0_s42"
FU2_RUN = "fu2_relabel_f1.0_s42"
ARMS = ["clean", "poisoned", "poisoned_repair", "poisoned_relabel"]
PER_CLASS = 300


def sha(p: Path) -> str:
    return S3.file_hash(p)


def first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0])
    return None


def find_src():
    src = X.find_dir(f"runs/cii/{RUN}/run_manifest.json", "NIDS_FU2_SRC")
    if src and not all((src / "runs" / "cii" / RUN / arm / "A2.pt").exists() for arm in ARMS[:3]):
        src = None  # folder found but without the deployed models
    work_run = Path(os.environ.get("NIDS_WORK", "/kaggle/working/nids_out")) / "runs" / "cii" / RUN
    if src is None and all((work_run / arm / "A2.pt").exists() for arm in ARMS[:3]):
        src = work_run.parents[2]  # rebuilt earlier in this session
    prep = X.find_dir("clean_test.parquet", "NIDS_PREPARED")
    return src, (src if src and (src / "clean_test.parquet").exists() else prep)


def find_fu2_model():
    rel = f"fu2/cii/{FU2_RUN}/poisoned_relabel/A2.pt"
    p = first([f"/kaggle/working/nids_out/{rel}", f"/kaggle/input/**/{rel}", f"./nids_out/{rel}"])
    if p:
        return p
    z = first(["/kaggle/input/**/fu2_restore.zip", "/kaggle/working/fu2_restore.zip"])
    if z:
        dest = Path("/kaggle/working/_fu2_restore") if Path("/kaggle/working").exists() else Path("_fu2_restore")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        return first([str(dest / "**" / FU2_RUN / "poisoned_relabel" / "A2.pt")])
    return None


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard_assets.json")
    ap.add_argument("--rebuild", action="store_true", help="re-run missing models with the committed code")
    a = ap.parse_args()

    src, prep = find_src()
    assert prep, "prepared data (clean_test.parquet, cii/scaler.json) not found: attach notebook 04's output"
    print("prepared data:", prep, "| sweep runs:", src)
    work = Path(os.environ.get("NIDS_WORK", "/kaggle/working/nids_out"))

    if src is None:
        assert a.rebuild, f"{RUN} not found; re-run with --rebuild (about 5 min on a T4)"
        if prep != work:  # s3 reads and writes one folder: copy the prepared (read-only) inputs into it
            import shutil
            work.mkdir(parents=True, exist_ok=True)
            for f in ["split_manifest.json", "class_map.json", "clean_train.parquet", "clean_val.parquet",
                      "clean_test.parquet", "protected_exclusions.json"]:
                if not (work / f).exists():
                    shutil.copy2(prep / f, work / f)
            if not (work / "cii").exists():
                shutil.copytree(prep / "cii", work / "cii")
            prep = work
        run("s3_continual.py", "--scenario", "cii", "--run", RUN, "--seed", 42, "--poison-frac", 1.0,
            "--forced-r1", "soft,threeway,threeway+relabel", "--resume")
        src = work
    rdir = src / "runs" / "cii" / RUN
    paths = {arm: rdir / arm / "A2.pt" for arm in ARMS[:3]}

    fu2 = find_fu2_model()
    if fu2 is None:
        assert a.rebuild, "FU2 seed-42 model not found; re-run with --rebuild (about 3 min on a T4)"
        run("s10_relabel.py", "run", "--src", src, "--work", work, "--seeds", 42)
        fu2 = work / "fu2" / "cii" / FU2_RUN / "poisoned_relabel" / "A2.pt"
    paths["poisoned_relabel"] = fu2
    committed = load_json(HERE / "experiments/results/06_fu2/fu2/cii" / FU2_RUN / "summary.json")
    fu2_same = sha(fu2) == committed["final_sha256"]
    print("FU2 model:", fu2, "| identical to the validated model (sha256):", fu2_same)

    hp = Namespace(**load_json(rdir / "config.json")["args"])
    ctx, *_ = S3.load_context(prep, "", "cii", "cpu")
    dev = torch.device("cpu")
    models = {arm: S3.load_ckpt(p, ctx, hp) for arm, p in paths.items()}

    # ---- the exported models must reproduce the committed full-test numbers
    ft = pd.read_csv(HERE / "experiments/results/04_sweep/final_test.csv")
    ft = ft[ft["run"] == RUN].set_index("arm")
    fu2_test = load_json(HERE / "experiments/results/06_fu2/test/fu2/cii" / FU2_RUN / "final_test.json")["test"]
    full, checks = {}, {}
    for arm, (m, cls) in models.items():
        met = S3.evaluate(m, cls, ctx["test"], dev, scope=ctx["scope"])
        full[arm] = {k: met[k] for k in ("n_rows", "rows_sha256", "accuracy", "macro_recall", "recall",
                                          "mean_conf", "benign_far", "n_benign", "confusion")}
        full[arm]["classes"] = cls
        want = fu2_test["recall"] if arm == "poisoned_relabel" else \
            {k: ft.loc[arm, f"recall_{k}"] for k in met["recall"]}
        diff = max(abs(met["recall"][k] - float(want[k])) for k in met["recall"])
        checks[arm] = {"max_abs_diff_vs_committed_test_recall": diff, "matches": diff < 1e-9}
        print(f"{arm:17s} test recall {({k: round(v, 4) for k, v in met['recall'].items()})} "
              f"| matches committed: {diff < 1e-9} (max diff {diff:.2e})")

    # ---- sample of raw test rows (same exclusions and scope as the scored test set)
    sc = load_json(prep / "cii" / "scaler.json")
    class_map = load_json(prep / "class_map.json")
    test_rids = set(ctx["test"].rid.tolist())
    df = pd.read_parquet(prep / "clean_test.parquet")
    df = df[df["_rid"].isin(test_rids)]
    rng = np.random.default_rng(2026)
    parts = []
    for k in ctx["scope"]:
        d = df[df[LABEL_COL] == k]
        parts.append(d.loc[np.sort(rng.choice(d.index.to_numpy(), size=min(PER_CLASS, len(d)), replace=False))])
    sample = pd.concat(parts).sort_values("_rid")
    rows = S3.Rows(sample, sc, class_map)
    ref = {}
    for arm, (m, cls) in models.items():
        P = S3.predict_proba(m, rows.x, rows.c, dev)
        ref[arm] = np.round(P.astype("float64"), 7).tolist()

    weights = {}
    for arm, (m, cls) in models.items():
        weights[arm] = {"classes": cls, "tensors": {k: {"shape": list(v.shape),
                                                        "data": v.detach().cpu().float().reshape(-1).tolist()}
                                                    for k, v in m.state_dict().items()}}
    cats = sc["features_categorical"]
    out = {
        "meta": {"setting": "CII, four classes, f = 1.0 (every stored Syn label flipped to BENIGN), seed 42",
                 "source_run": RUN, "fu2_run": FU2_RUN, "fu2_model_is_validated_model": fu2_same,
                 "model_sha256": {arm: sha(p) for arm, p in paths.items()},
                 "full_test_checks": checks, "sample": f"{PER_CLASS} test rows per class, numpy seed 2026",
                 "hp": {k: getattr(hp, k) for k in ("d", "heads", "layers")},
                 "git": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip(),
                 "torch": torch.__version__},
        "scaler": {"features_numeric": sc["features_numeric"], "mean": sc["mean"], "std": sc["std"],
                   "clip": sc["clip"], "features_categorical": cats, "cat_vocab": sc["cat_vocab"]},
        "rows": {"rid": rows.rid.tolist(), "label": rows.lab.tolist(),
                 "num": sample.sort_values("_rid")[sc["features_numeric"]].to_numpy("float64").tolist(),
                 "cat": sample.sort_values("_rid")[cats].astype(int).to_numpy().tolist()},
        "ref": ref, "full_test": full, "models": weights,
    }
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out}: {Path(a.out).stat().st_size / 1e6:.1f} MB | rows {len(rows.rid)} | "
          f"all four models match the committed test numbers: {all(c['matches'] for c in checks.values())}")


if __name__ == "__main__":
    main()
