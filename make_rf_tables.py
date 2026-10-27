"""RF held-out test tables for git, from SAVED files only (trains and scores nothing).

  python make_rf_tables.py --test <extracted rf_test.zip>/rf/test

RF (post hoc; fixed-class traffic refresh, NOT new-class arrival): every arm's deployed A0..A6 was scored once by
`s14_refresh.py score-test` at decisions commit b0d6b63. This script checks that every test file records that
commit, that all models were scored on the same rows, that test_trajectory.csv equals the JSON files, and that the
committed validation summary lists the same runs; then it copies the files unchanged and writes the tables.

Writes to experiments/results/10_rf/test/:
  rf/test/<run>/<arm>.json, rf/test/test_trajectory.csv   copied unchanged
  test_final.csv     per rho, arm and seed: the deployed model after E6 (A6): Syn / TFTP / BENIGN recall, Syn mean p,
                     benign FAR, macro recall, cumulative flipped share of the stored Syn rows
  test_summary.csv   per rho and arm: mean, min and max over seeds 42-44 of the same metrics, plus A0 for reference
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import load_json

VAL = Path("experiments/results/10_rf")
DECISIONS_COMMIT = "b0d6b636a93bba876922aa5774e93406a524d6d8"
ARMS = ("clean", "poisoned", "poisoned_repair")
METRICS = ["Syn_recall", "TFTP_recall", "BENIGN_recall", "Syn_mean_p", "benign_far", "macro_recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="rf/test/ folder of rf_test.zip")
    ap.add_argument("--dest", default=str(VAL / "test"))
    a = ap.parse_args()
    src, dest = Path(a.test), Path(a.dest)
    summ = load_json(VAL / "rf" / "summary.json")  # the committed validation summary
    rows, hashes = [], set()
    for name, rs in summ["runs"].items():
        for arm in ARMS:
            j = load_json(src / name / f"{arm}.json")
            assert j["arm"] == arm and j["decisions_commit"] == DECISIONS_COMMIT, f"{name}/{arm}: other commit"
            assert [e["t"] for e in j["trajectory"]] == list(range(7))
            for e in j["trajectory"]:
                te = e["test"]
                hashes.add(te["rows_sha256"])
                rows.append({"run": name, "rho": rs["rho"], "seed": rs["seed"], "arm": arm, "t": e["t"],
                             "flipped_share": 0.0 if arm == "clean" else rs["rho"] * e["t"],
                             **{f"{k}_recall": te["recall"].get(k, 0.0) for k in ("Syn", "TFTP", "BENIGN")},
                             "Syn_mean_p": te["mean_conf"]["Syn"], "benign_far": te["benign_far"],
                             "macro_recall": te["macro_recall"], "n_rows": te["n_rows"], "n_benign": te["n_benign"],
                             "test_rows_sha256": te["rows_sha256"]})
            (dest / "rf" / "test" / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(src / name / f"{arm}.json", dest / "rf" / "test" / name / f"{arm}.json")
    assert len(hashes) == 1, "different test rows across models"
    df = pd.DataFrame(rows)
    tc = pd.read_csv(src / "test_trajectory.csv").merge(df, on=["run", "arm", "t"], suffixes=("", "_json"))
    assert len(tc) == len(df) == 6 * 3 * 7
    for c in ("Syn_recall", "Syn_mean_p", "benign_far", "macro_recall"):
        assert (tc[c] - tc[f"{c}_json"]).abs().max() < 1e-12, f"test_trajectory.csv differs from the JSON files: {c}"
    shutil.copy(src / "test_trajectory.csv", dest / "rf" / "test" / "test_trajectory.csv")

    final = df[df["t"] == 6].drop(columns=["run", "t"]).sort_values(["rho", "arm", "seed"])
    final.to_csv(dest / "test_final.csv", index=False)
    out = []
    for (rho, arm), g in df.groupby(["rho", "arm"]):
        for t in (0, 6):
            x = g[g["t"] == t]
            out.append({"rho": rho, "arm": arm, "model": f"A{t}", "flipped_share": x["flipped_share"].iloc[0],
                        **{f"{m}_{s}": getattr(x[m], s)() for m in METRICS for s in ("mean", "min", "max")}})
    sm = pd.DataFrame(out)
    sm.to_csv(dest / "test_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"test rows: {df['n_rows'].iloc[0]} ({df['n_benign'].iloc[0]} benign), one row hash; "
          f"decisions commit {DECISIONS_COMMIT[:7]} in all {len(df) // 7} files")
    show = ["rho", "arm", "model", "flipped_share"] + [f"{m}_mean" for m in METRICS]
    print(sm[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
