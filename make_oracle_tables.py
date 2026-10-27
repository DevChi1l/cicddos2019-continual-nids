"""Joint-oracle test tables for git, from SAVED files only (trains and scores nothing).

  python make_oracle_tables.py --oracle <extracted oracle_test.zip>/oracle

The oracle is ONE model trained jointly on all 12 labels (no updates, gate or replay; `s11_oracle.py`). It is not
a continual model. Its test numbers are shown next to the continual models of the committed tables:
`04b_all12/final_test.csv` (s3 clean arm) and `05_fu1/test/test_recall.csv` (FU1), on the same test rows (row sha256
asserted equal). The zip's val.json / oracle_val.csv / verdict.json must equal the committed validation copies.

Writes to experiments/results/07_oracle/test/:
  oracle/<sc>/<setting>_s<seed>/test.json   copied unchanged
  test_recall.csv     per cell: recall per class, macro recall, benign FAR, pair confusion shares
  test_median.csv     CII median over seeds 42-44 per setting (the registered primary data), and the CI cells
  test_vs_continual.csv  per scenario and class: oracle (CII median / CI seed 42, both settings) vs s3 clean and FU1
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import load_json

VAL = Path("experiments/results/07_oracle")
CELLS = [("cii", s, seed) for s in ("e10", "e20") for seed in (42, 43, 44)] + [("ci", s, 42) for s in ("e10", "e20")]
PAIRS = [("DNS", "LDAP"), ("SSDP", "UDP")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="oracle/ folder of oracle_test.zip")
    ap.add_argument("--dest", default=str(VAL / "test"))
    a = ap.parse_args()
    src, dest = Path(a.oracle), Path(a.dest)
    for f in ["oracle_val.csv", "verdict.json"] + [f"{sc}/{s}_s{seed}/val.json" for sc, s, seed in CELLS]:
        assert (src / f).read_bytes() == (VAL / "oracle" / f).read_bytes(), f"{f}: not the committed validation file"
    s3 = pd.read_csv("experiments/results/04b_all12/final_test.csv")
    fu1 = pd.read_csv("experiments/results/05_fu1/test/test_recall.csv")
    fu1s = pd.read_csv("experiments/results/05_fu1/test/test_summary.csv")

    rows = []
    for sc, setting, seed in CELLS:
        rel = Path(sc) / f"{setting}_s{seed}"
        j = load_json(src / rel / "test.json")
        t = j["test"]
        c = s3[(s3["scenario"] == sc) & (s3["arm"] == "clean")].iloc[0]
        assert t["rows_sha256"] == c["test_rows_sha256"] and t["n_rows"] == c["test_rows"], f"{rel}: different test rows"
        rows.append({"scenario": sc, "setting": setting, "seed": seed, "decisions_commit": j["decisions_commit"],
                     **{k: t["recall"][k] for k in t["support"]}, "macro_recall": t["macro_recall"],
                     "benign_far": t["benign_far"], "n_benign": t["n_benign"],
                     **{f"{x}_to_{y}": t["confusion"][x].get(y, 0) / t["support"][x] for p in PAIRS
                        for x, y in (p, p[::-1])}})
        (dest / "oracle" / rel).mkdir(parents=True, exist_ok=True)
        shutil.copy(src / rel / "test.json", dest / "oracle" / rel / "test.json")
    df = pd.DataFrame(rows)
    df.to_csv(dest / "test_recall.csv", index=False)
    classes = list(load_json(src / CELLS[0][0] / f"{CELLS[0][1]}_s{CELLS[0][2]}" / "test.json")["test"]["support"])
    num = classes + ["macro_recall", "benign_far"] + [c for c in df.columns if "_to_" in c]
    med = df[df["scenario"] == "cii"].groupby("setting")[num].median().reset_index()
    med.insert(0, "scenario", "cii (median of 3 seeds)")
    ci = df[df["scenario"] == "ci"][["scenario", "setting"] + num].assign(scenario="ci (seed 42)")
    pd.concat([med, ci]).to_csv(dest / "test_median.csv", index=False)

    cmp = []
    for sc, lab in (("cii", "cii (median of 3 seeds)"), ("ci", "ci (seed 42)")):
        o = pd.concat([med, ci]).set_index(["scenario", "setting"]).loc[lab]
        c = s3[(s3["scenario"] == sc) & (s3["arm"] == "clean")].iloc[0]
        f = fu1[fu1["scenario"] == sc].set_index("class")
        fs = fu1s[(fu1s["scenario"] == sc) & (fu1s["model"] == "fu1")].iloc[0]
        for k in classes + ["macro_recall", "benign_far"]:
            whole = k in ("macro_recall", "benign_far")
            cmp.append({"scenario": sc, "metric": k if whole else f"recall_{k}",
                        "oracle_joint_e10": o.loc["e10", k], "oracle_joint_e20": o.loc["e20", k],
                        "continual_s3_clean": c[f"test_{k}"] if whole else c[f"test_recall_{k}"],
                        "continual_fu1": fs["macro_recall_12" if k == "macro_recall" else k] if whole else f.loc[k, "fu1"]})
    pd.DataFrame(cmp).to_csv(dest / "test_vs_continual.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[["scenario", "setting", "seed", *classes, "macro_recall", "benign_far"]].round(3).to_string(index=False))
    print(pd.DataFrame(cmp).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
