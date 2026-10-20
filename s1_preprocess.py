"""Stage 1 - Global dedupe, then cap, then split: flow files -> protected, row-ID'd splits (UNSCALED).

CICDDoS2019 is mostly copies (the full audit found ~59.9M within-file duplicate rows of 70.4M), so
capping must happen AFTER deduplication. One streaming pass, bounded memory:

  1. Read each file in chunks; canonical labels from the LABEL COLUMN (filenames lie on 03-11).
  2. inf -> NaN; rows with NaN features are dropped (--nonfinite drop) or zero-filled (zero).
  3. Feature hash h = hash of the model features only (ids, IPs, ports, timestamp, Inbound
     excluded; all cast to float64 so the hash does not depend on per-chunk dtypes).
     Identical h = identical model input, in any file.
  4. Every distinct (h, label) pair is recorded -> exact distinct counts per label, and the set
     of h seen under 2+ labels (label conflicts, all dropped: we cannot know which is right).
  5. Bottom-k sampling: per label keep the --cap distinct rows with the smallest h. Because h is a
     hash, this is a uniform, deterministic sample of DISTINCT rows, independent of file order.
     BENIGN is never capped: every distinct benign row is kept.
  6. Each surviving feature vector occurs exactly once (asserted), gets a row id (_rid), and is
     split per class into train / calib / val / test. No model-input vector can sit in 2 splits.
  7. Saved UNSCALED; the scaler is fit later on E0 train rows (Stage 1b).
Row counts read are checked against the full audit manifest (must be untruncated).

Usage:
  python s1_preprocess.py --input /kaggle/input --cap 60000 --split random
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CATEGORICAL_COLS, DROP_INBOUND, DROP_PORTS, ID_COLS, LABEL_COL,
                    LEAK_COLS, PORT_COLS, SEED, TIME_COL, canon_label, default_work_dir,
                    find_files, iter_chunks, load_json, save_json)

SPLIT_FRACS = {"train": 0.60, "calib": 0.10, "val": 0.10, "test": 0.20}


def feature_hash(df: pd.DataFrame, feat: list[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(df[feat].astype("float64"), index=False).to_numpy()


def stream_dedupe_cap(files, cap, chunksize, nonfinite):
    """One pass. Returns (kept rows, feature lists, stats)."""
    feat = num_cols = cat_cols = None
    kept: dict[str, pd.DataFrame] = {}
    pairs_h, pairs_l, label_ids = [], [], {}
    raw, nonfin, rows_by_file = Counter(), Counter(), {}
    t0 = time.time()
    for p in files:
        n_file = 0
        for df in iter_chunks(p, chunksize):
            if LABEL_COL not in df.columns:
                continue
            if feat is None:  # fix the model feature set from the first chunk
                drop = set(ID_COLS) | (PORT_COLS if DROP_PORTS else set()) | (LEAK_COLS if DROP_INBOUND else set())
                cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
                skip = drop | {LABEL_COL} | set(cat_cols)
                num_cols = [c for c in df.columns
                            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
                feat = num_cols + cat_cols
            n_file += len(df)
            df[LABEL_COL] = df[LABEL_COL].map(canon_label)
            raw.update(df[LABEL_COL].tolist())
            X = df[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            bad = X.isna().any(axis=1).to_numpy()
            nonfin.update(df.loc[bad, LABEL_COL].tolist())
            if nonfinite == "drop":
                df, X = df.loc[~bad], X.loc[~bad]
            else:
                X[num_cols] = X[num_cols].fillna(0.0)
                ok = ~X[cat_cols].isna().any(axis=1).to_numpy() if cat_cols else np.ones(len(X), bool)
                df, X = df.loc[ok], X.loc[ok]
            keep_cols = [c for c in [TIME_COL] if c in df.columns]
            part = pd.concat([X, df[[LABEL_COL] + keep_cols]], axis=1)
            part["_src"] = f"{p.parent.name}/{p.name}"  # day folder + file = provenance
            part["_h"] = feature_hash(part, feat)
            part = part.drop_duplicates(["_h", LABEL_COL])
            lid = part[LABEL_COL].map(lambda s: label_ids.setdefault(s, len(label_ids))).to_numpy()
            pairs_h.append(part["_h"].to_numpy())
            pairs_l.append(lid.astype("int16"))
            for lab, g in part.groupby(LABEL_COL):
                cur = kept.get(lab)
                if lab != "BENIGN" and cur is not None and len(cur) >= cap:
                    g = g[g["_h"] < cur["_h"].max()]
                    if g.empty:
                        continue
                cur = g if cur is None else pd.concat([cur, g], ignore_index=True)
                cur = cur.drop_duplicates("_h")  # first file seen keeps provenance
                if lab != "BENIGN" and len(cur) > cap:
                    cur = cur.nsmallest(cap, "_h")
                kept[lab] = cur
        rows_by_file[str(p)] = n_file
        print(f"  {p.parent.name}/{p.name}: {n_file:,} rows  ({time.time() - t0:.0f}s elapsed)", flush=True)

    # exact distinct counts and label conflicts from all (h, label) pairs
    pairs = pd.DataFrame({"h": np.concatenate(pairs_h), "l": np.concatenate(pairs_l)}).drop_duplicates()
    names = {v: k for k, v in label_ids.items()}
    distinct = pairs["l"].map(names).value_counts().to_dict()
    nlab = pairs.groupby("h")["l"].nunique()
    conflict_h = nlab.index[nlab > 1].to_numpy()
    conflict_by_label = pairs[pairs["h"].isin(conflict_h)]["l"].map(names).value_counts().to_dict()
    del pairs, pairs_h, pairs_l

    df = pd.concat(kept.values(), ignore_index=True)
    before = df[LABEL_COL].value_counts().to_dict()
    df = df[~df["_h"].isin(conflict_h)].reset_index(drop=True)
    stats = {
        "seconds_pass": round(time.time() - t0),
        "raw_rows_by_label": dict(raw), "nonfinite_rows_by_label": dict(nonfin),
        "nonfinite_policy": nonfinite,
        "distinct_rows_by_label": {k: int(v) for k, v in distinct.items()},
        "conflict_distinct_by_label": {k: int(v) for k, v in conflict_by_label.items()},
        "n_conflict_vectors": int(len(conflict_h)),
        "sampled_before_conflict_drop": {k: int(v) for k, v in before.items()},
        "kept_by_label": {k: int(v) for k, v in df[LABEL_COL].value_counts().items()},
        "benign_distinct_kept": int((df[LABEL_COL] == "BENIGN").sum()),
        "cap_distinct_per_attack_class": cap,
        "rows_by_file": rows_by_file,
    }
    return df, num_cols, cat_cols, stats


def split_indices(df, mode, rng):
    """Series of split names, assigned per class so rare classes appear everywhere."""
    out = pd.Series("", index=df.index, dtype=object)
    ts = pd.to_datetime(df[TIME_COL], errors="coerce") if TIME_COL in df.columns else None
    notes = {}
    for lab, g in df.groupby(LABEL_COL):
        idx = g.index.to_numpy()
        if mode == "time" and ts is not None and ts.loc[idx].notna().all():
            idx = ts.loc[idx].sort_values(kind="stable").index.to_numpy()
            notes[lab] = "time-ordered"
        else:
            idx = rng.permutation(idx)
            notes[lab] = "random" + (" (no usable timestamps)" if mode == "time" else "")
        n = len(idx)
        a = int(n * SPLIT_FRACS["train"])
        b = a + int(n * SPLIT_FRACS["calib"])
        c = b + int(n * SPLIT_FRACS["val"])
        out.loc[idx[:a]] = "train"
        out.loc[idx[a:b]] = "calib"
        out.loc[idx[b:c]] = "val"
        out.loc[idx[c:]] = "test"
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=int, default=60_000, help="max DISTINCT rows kept per attack class")
    ap.add_argument("--split", choices=["time", "random"], default="random")
    ap.add_argument("--nonfinite", choices=["drop", "zero"], default="drop",
                    help="inf/NaN features: drop the row, or set to 0 (what the ReplayIDS release does)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    a = ap.parse_args()
    out = default_work_dir() if a.out is None else Path(a.out)
    rng = np.random.default_rng(SEED)

    manifest = load_json(out / "manifest.json")  # full audit from s0_audit.py
    if any(f["scan_truncated"] for f in manifest["files"]):
        raise SystemExit("manifest.json came from a truncated audit; rerun s0_audit.py without "
                         "--max-rows-per-file.")
    files = find_files(a.input)
    audited = {f["path"]: f["rows_scanned"] for f in manifest["files"]}
    if sorted(map(str, files)) != sorted(audited):
        raise SystemExit("files under --input differ from the audited files; rerun the audit.")

    df, num_cols, cat_cols, stats = stream_dedupe_cap(files, a.cap, a.chunksize, a.nonfinite)
    mismatch = {p: (n, audited[p]) for p, n in stats["rows_by_file"].items() if n != audited[p]}
    if mismatch:
        raise SystemExit(f"rows read differ from the audit (read, audited): {mismatch}")
    if not df["_h"].is_unique:
        raise SystemExit("a model-feature vector survived twice; refusing to split")

    df = df.sort_values("_h").reset_index(drop=True)  # order independent of file order
    df["_rid"] = np.arange(len(df), dtype="int64")
    df["_split"], split_notes = split_indices(df, a.split, rng)
    per_split_h = [set(df.loc[df["_split"] == s, "_h"]) for s in SPLIT_FRACS]
    overlap = sum(len(x & y) for i, x in enumerate(per_split_h) for y in per_split_h[i + 1:])
    if overlap:
        raise SystemExit(f"{overlap} identical model-feature vectors in two splits; refusing")

    labels = sorted(df[LABEL_COL].unique(), key=lambda s: (s != "BENIGN", s))
    class_map = {lab: i for i, lab in enumerate(labels)}
    counts = {}
    keep_cols = ["_rid", "_h", "_src", LABEL_COL] + num_cols + cat_cols + \
        ([TIME_COL] if TIME_COL in df.columns else [])
    for sp in SPLIT_FRACS:
        part = df.loc[df["_split"] == sp, keep_cols]
        part.to_parquet(out / f"clean_{sp}.parquet", index=False)
        counts[sp] = part[LABEL_COL].value_counts().to_dict()

    save_json({"numeric": num_cols, "categorical": cat_cols}, out / "features.json")
    save_json(class_map, out / "class_map.json")
    save_json({
        "args": vars(a), "seed": SEED, "split_fracs": SPLIT_FRACS,
        "split_mode_per_class": split_notes, "stats": stats,
        "counts_per_split": counts, "cross_split_feature_overlap": overlap,
        "dedupe": "global, on model features (float64 hash), before capping; label conflicts dropped",
        "note": ("Random flow-level splits can leak near-identical neighbouring flows across "
                 "splits; exact duplicates cannot. Neither is a deployment-chronology claim."),
    }, out / "split_manifest.json")

    print(f"\nModel features: {len(num_cols)} numeric + {cat_cols}")
    print(f"Pass took {stats['seconds_pass']}s")
    tab = pd.DataFrame({"raw": stats["raw_rows_by_label"],
                        "nonfinite": stats["nonfinite_rows_by_label"],
                        "distinct": stats["distinct_rows_by_label"],
                        "conflict": stats["conflict_distinct_by_label"],
                        "kept": stats["kept_by_label"]}).fillna(0).astype(int)
    print(tab.sort_values("raw", ascending=False))
    print(f"\nDISTINCT BENIGN KEPT: {stats['benign_distinct_kept']:,}")
    print("\nPer-split class counts:")
    print(pd.DataFrame(counts).fillna(0).astype(int))
    print(f"\ncross-split identical feature vectors: {overlap}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
