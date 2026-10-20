"""Stage 0 - Dataset audit. Run this BEFORE any training.

Scans every CSV/parquet under --input and writes manifest.json with, per file:
  columns, row count, canonical label counts, non-finite counts, timestamp
  range, exact-duplicate-row count, (optionally) sha256.
It also prints warnings: unknown labels, schema drift between files,
constant columns, label/ID-like columns that could leak.

Usage (Kaggle):
  !python s0_audit.py --input /kaggle/input --hash
Quick look at big files:
  !python s0_audit.py --input /kaggle/input --max-rows-per-file 2000000
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from common import (CATEGORICAL_COLS, ID_COLS, LABEL_COL, LEAK_COLS, PORT_COLS, TIME_COL, canon_label,
                    default_work_dir, find_files, is_known_label, iter_chunks,
                    save_json, sha256_file)


def audit_file(path, chunksize, max_rows, do_hash):
    info = {"path": str(path), "bytes": path.stat().st_size}
    if do_hash:
        info["sha256"] = sha256_file(path)
    labels = Counter()
    n = 0
    nonfinite = Counter()
    nonfinite_rows_by_label = Counter()  # rows with ANY inf/NaN, per label
    rows_by_label_scanned = Counter()
    const_tracker: dict[str, set] = {}
    tmin = tmax = None
    inbound = Counter()  # (label, inbound value) -> rows, to test for leakage
    hashes = []
    cols = None
    dtypes = {}
    for df in iter_chunks(path, chunksize):
        if cols is None:
            cols = list(df.columns)
            dtypes = {c: str(t) for c, t in df.dtypes.items()}
        if max_rows and n >= max_rows:
            break
        if max_rows:
            df = df.iloc[: max_rows - n]
        n += len(df)
        if LABEL_COL in df.columns:
            labels.update(df[LABEL_COL].map(canon_label).tolist())
            if "inbound" in df.columns:
                pair = zip(df[LABEL_COL].map(canon_label), df["inbound"])
                inbound.update((f"{l}|{v}" for l, v in pair))
        num = df.select_dtypes(include=[np.number])
        arr = num.to_numpy(dtype="float64", copy=False)
        bad = ~np.isfinite(arr)
        if LABEL_COL in df.columns:
            lab = df[LABEL_COL].map(canon_label)
            nonfinite_rows_by_label.update(lab[bad.any(axis=1)].tolist())
        for j, c in enumerate(num.columns):
            k = int(bad[:, j].sum())
            if k:
                nonfinite[c] += k
        for c in num.columns:  # cheap constant-column tracking
            s = const_tracker.setdefault(c, set())
            if len(s) < 2:
                s.update(num[c].dropna().unique()[:2].tolist())
        if TIME_COL in df.columns:
            ts = pd.to_datetime(df[TIME_COL], errors="coerce")
            if ts.notna().any():
                lo, hi = ts.min(), ts.max()
                tmin = lo if tmin is None else min(tmin, lo)
                tmax = hi if tmax is None else max(tmax, hi)
        if len(hashes) * chunksize < 20_000_000:  # cap memory
            # Hash features + label only. Hashing ids (unnamed: 0, flow id) made every
            # row unique, so the old duplicate count was always 0.
            skip = ID_COLS | PORT_COLS | LEAK_COLS
            keep = [c for c in df.columns if c not in skip]
            hashes.append(pd.util.hash_pandas_object(df[keep], index=False).to_numpy())
    dup = None
    if hashes:
        h = np.concatenate(hashes)
        dup = int(len(h) - len(np.unique(h)))
    info.update({
        "rows_scanned": n,
        "scan_truncated": bool(max_rows and n >= max_rows),
        "columns": cols,
        "dtypes": dtypes,
        "label_counts": dict(labels),
        "inbound_by_label": dict(inbound),
        "nonfinite_by_col": dict(nonfinite),
        "nonfinite_rows_by_label": dict(nonfinite_rows_by_label),
        "constant_cols": sorted(c for c, s in const_tracker.items() if len(s) <= 1),
        "time_min": str(tmin), "time_max": str(tmax),
        "exact_duplicate_rows": dup,
    })
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--max-rows-per-file", type=int, default=0, help="0 = scan all")
    ap.add_argument("--hash", action="store_true", help="sha256 every file (slow on big data)")
    a = ap.parse_args()

    out = (default_work_dir() if a.out is None else __import__("pathlib").Path(a.out))
    files = find_files(a.input)
    if not files:
        raise SystemExit(f"No .csv/.parquet files under {a.input}")
    print(f"Found {len(files)} files")

    manifest = {"input_root": a.input, "files": []}
    for p in files:
        print(f"... auditing {p.name}")
        manifest["files"].append(audit_file(p, a.chunksize, a.max_rows_per_file, a.hash))

    # ---- cross-file summary and warnings ----
    total = Counter()
    for f in manifest["files"]:
        total.update(f["label_counts"])
    manifest["label_totals"] = dict(total)
    benign_by_file = {f["path"]: f["label_counts"].get("BENIGN", 0) for f in manifest["files"]}
    manifest["benign_by_file"] = benign_by_file
    nfl = Counter()
    for f in manifest["files"]:
        nfl.update(f.get("nonfinite_rows_by_label", {}))
    manifest["nonfinite_rows_by_label"] = dict(nfl)
    inb = Counter()
    for f in manifest["files"]:
        inb.update(f.get("inbound_by_label", {}))
    manifest["inbound_by_label"] = dict(inb)

    warnings = []
    unknown = [l for l in total if not is_known_label(l)]
    if unknown:
        warnings.append(f"Unknown label spellings (add to _CANON in common.py): {unknown}")
    colsets = [tuple(f["columns"]) for f in manifest["files"] if f["columns"]]
    if len(set(colsets)) > 1:
        warnings.append("Schema differs between files (column names/order). Inspect before concatenating.")
    for f in manifest["files"]:
        if LABEL_COL not in (f["columns"] or []):
            warnings.append(f"{f['path']}: no '{LABEL_COL}' column found.")
        if f["exact_duplicate_rows"]:
            warnings.append(f"{f['path']}: {f['exact_duplicate_rows']} exact duplicate rows in scanned part.")
        if f["nonfinite_by_col"]:
            warnings.append(f"{f['path']}: non-finite values in {len(f['nonfinite_by_col'])} columns.")
    if any(f["scan_truncated"] for f in manifest["files"]):
        warnings.append("Scan was truncated (--max-rows-per-file): counts are a PREFIX only. "
                        "Do not feed this manifest to s1_preprocess.py; rerun without the limit.")
    first_cols = manifest["files"][0]["columns"] or []
    ids_present = sorted(c for c in first_cols if c in ID_COLS)
    manifest["id_like_columns_present"] = ids_present
    manifest["categorical_candidates_present"] = [c for c in CATEGORICAL_COLS if c in first_cols]
    manifest["warnings"] = warnings
    save_json(manifest, out / "manifest.json")

    print("\n=== LABEL TOTALS (canonical) ===")
    for l, c in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"{l:>12}: {c:>12,}")
    print("\n=== BENIGN ROWS PER FILE (raw, before dedupe) ===")
    for pth, c in benign_by_file.items():
        print(f"{c:>12,}  {pth}")
    print(f"{sum(benign_by_file.values()):>12,}  TOTAL benign "
          f"({100 * sum(benign_by_file.values()) / max(1, sum(total.values())):.3f}% of scanned rows)")
    print("\n=== ROWS WITH inf/NaN, PER LABEL (dropping them costs this much; benign is scarce) ===")
    for l, c in sorted(total.items(), key=lambda kv: -kv[1]):
        k = nfl.get(l, 0)
        print(f"{l:>12}: {k:>10,} of {c:>12,}  ({100 * k / max(1, c):5.2f}%)")
    if inb:
        print("\n=== INBOUND x LABEL (leak check: a label living on one value is a giveaway) ===")
        for k, c in sorted(inb.items()):
            print(f"{k:>24}: {c:>12,}")
    print("\n=== PER FILE ===")
    for f in manifest["files"]:
        print(f"{f['path']}\n   rows={f['rows_scanned']:,} cols={len(f['columns'] or [])} "
              f"time=[{f['time_min']} .. {f['time_max']}] labels={f['label_counts']}")
    print("\n=== WARNINGS ===")
    print("\n".join(warnings) if warnings else "none")
    print(f"\nManifest written to {out/'manifest.json'}")


if __name__ == "__main__":
    main()
