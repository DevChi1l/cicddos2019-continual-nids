"""All-labels schedule, derived by a fixed rule from split_manifest.json (no training, no data read).

Rule (written before any all-labels run):
  1. An attack class is included if it has >= MIN_TRAIN train rows and >= MIN_VAL val rows
     (before Stage 1b exclusions). MIN_VAL = 200 so that the gate tolerance of 0.05 is >= 10 rows.
  2. E0 = BENIGN + Syn, as in `main`, so the teacher T has the same job.
  3. The remaining classes, sorted by name, arrive two per experience.
  4. Attack rows per class per experience = the smallest train count among included classes
     (pass it to s1b as --attack-cap), so every class contributes the same number of rows.
  5. Benign: CII cuts the train benign rows into k disjoint near-equal portions (s1b
     build_manifest); CI puts every train benign row in E0.

  python make_all_labels_schedule.py --split-manifest nids_out/split_manifest.json --write all12.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_TRAIN, MIN_VAL, E0 = 1000, 200, "Syn"


def derive(split_manifest: dict) -> dict:
    tr, va = split_manifest["counts_per_split"]["train"], split_manifest["counts_per_split"]["val"]
    attacks = sorted(k for k in tr if k != "BENIGN")
    keep = [k for k in attacks if tr[k] >= MIN_TRAIN and va.get(k, 0) >= MIN_VAL]
    dropped = {k: {"train": tr[k], "val": va.get(k, 0)} for k in attacks if k not in keep}
    assert E0 in keep, f"{E0} must be included"
    rest = [k for k in keep if k != E0]
    schedule = [[E0]] + [rest[i:i + 2] for i in range(0, len(rest), 2)]
    cap = min(tr[k] for k in keep)
    k = len(schedule)
    nb = tr["BENIGN"]
    return {"experiences": schedule, "attack_cap": cap, "labels_used": 1 + len(keep),
            "labels_total": len(tr), "excluded": dropped,
            "rule": {"min_train": MIN_TRAIN, "min_val": MIN_VAL, "e0": E0, "per_experience": 2},
            "cii_benign_per_experience": [nb // k + (1 if i < nb % k else 0) for i in range(k)],
            "ci_benign": [nb] + [0] * (k - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--write", default=None, help="schedule JSON for s1b --schedule")
    a = ap.parse_args()
    d = derive(json.loads(Path(a.split_manifest).read_text()))
    print(json.dumps({k: v for k, v in d.items()}, indent=1))
    if a.write:
        Path(a.write).write_text(json.dumps(d, indent=1))  # s1b reads only "experiences"
        print(f"written {a.write}; run s1b with --attack-cap {d['attack_cap']}")


if __name__ == "__main__":
    main()
