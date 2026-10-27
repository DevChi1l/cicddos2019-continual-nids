"""FU1 test tables for git, from SAVED files only (trains and scores nothing).

  python make_fu1_tables.py --fu1 <extracted fu1_test.zip>/followup --src <nids_out of notebook 04>

Compares the FU1 deployed model (`s9_followup.py score-test`) with the s3 clean arm of the same all12 run on the
same test rows (row sha256 asserted equal). The s3 numbers come from the source run's clean/final_test.json and
are checked against the committed experiments/results/04b_all12/final_test.csv.

Writes to experiments/results/05_fu1/test/:
  followup/<sc>/fu1_retry_<sc>_s42/final_test.json   copied unchanged
  test_recall.csv        per scenario and class: support, s3 clean recall, FU1 recall
  test_summary.csv       per scenario and model: labels learned, macro recall, benign FAR, attack-not-benign rate
  test_unlearned.csv     FU1: where the test rows of each class it did not learn were predicted

"attack_not_benign" (share of attack test rows predicted as ANY attack class) is a post-hoc descriptive number,
not in the pre-registration; it says how often an attack is still flagged even when its label is wrong.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from common import load_json


def attack_not_benign(t: dict) -> float:
    rows = [(k, v) for k, v in t["confusion"].items() if k != "BENIGN"]
    return sum(sum(v.values()) - v.get("BENIGN", 0) for _, v in rows) / sum(sum(v.values()) for _, v in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fu1", required=True, help="followup/ folder holding fu1_retry_<sc>_s42/final_test.json")
    ap.add_argument("--src", required=True, help="nids_out of notebook 04 (runs/<sc>/all12_<sc>_s42/clean/)")
    ap.add_argument("--dest", default="experiments/results/05_fu1/test")
    a = ap.parse_args()
    fu1, src, dest = Path(a.fu1), Path(a.src), Path(a.dest)
    committed = pd.read_csv("experiments/results/04b_all12/final_test.csv")

    rec, summ, unl = [], [], []
    for sc in ("cii", "ci"):
        rel = Path(sc) / f"fu1_retry_{sc}_s42" / "final_test.json"
        f = load_json(fu1 / rel)
        s = load_json(src / "runs" / sc / f"all12_{sc}_s42" / "clean" / "final_test.json")
        ft, st = f["test"], s["test"]
        assert ft["rows_sha256"] == st["rows_sha256"] and ft["n_rows"] == st["n_rows"], f"{sc}: different test rows"
        assert ft["support"] == st["support"]
        c = committed[(committed["scenario"] == sc) & (committed["arm"] == "clean")].iloc[0]
        assert c["test_rows_sha256"] == st["rows_sha256"]
        assert all(abs(c[f"test_recall_{k}"] - v) < 1e-12 for k, v in st["recall"].items()), "04b_all12 table differs"
        (dest / "followup" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fu1 / rel, dest / "followup" / rel)

        for k, n in ft["support"].items():
            rec.append({"scenario": sc, "class": k, "test_rows": n, "s3_clean": st["recall"][k],
                        "fu1": ft["recall"][k], "change": ft["recall"][k] - st["recall"][k]})
        for name, d, t in (("s3_clean", s, st), ("fu1", f, ft)):
            summ.append({"scenario": sc, "model": name, "labels_learned": len(d["final_classes"]),
                         "classes": " ".join(d["final_classes"]), "macro_recall_12": t["macro_recall"],
                         "accuracy": t["accuracy"], "benign_far": t["benign_far"], "n_benign": t["n_benign"],
                         "new_class_far": t["new_class_far"], "attack_not_benign": attack_not_benign(t),
                         "test_rows": t["n_rows"], "test_rows_sha256": t["rows_sha256"],
                         "decisions_commit": d.get("decisions_commit", "")})
        for k, row in ft["confusion"].items():
            if k in f["final_classes"]:
                continue
            n = sum(row.values())
            top = max(row, key=row.get)
            unl.append({"scenario": sc, "class": k, "test_rows": n, "top_prediction": top,
                        "top_share": row[top] / n, "pred_benign_share": row.get("BENIGN", 0) / n,
                        "predicted": json.dumps(row, sort_keys=True)})

    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rec).to_csv(dest / "test_recall.csv", index=False)
    pd.DataFrame(summ).to_csv(dest / "test_summary.csv", index=False)
    pd.DataFrame(unl).to_csv(dest / "test_unlearned.csv", index=False)
    print(pd.DataFrame(summ).drop(columns=["classes", "test_rows_sha256", "decisions_commit"]).round(4).to_string(index=False))
    print(f"written to {dest}")


if __name__ == "__main__":
    main()
