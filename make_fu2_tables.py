"""FU2 test tables for git, from SAVED files only (trains and scores nothing).

  python make_fu2_tables.py --fu2 <extracted fu2_test.zip>/fu2 --src <nids_out of notebook 04>

Compares the FU2 deployed model (`s10_relabel.py score-test`, arm poisoned_relabel) with the sweep's clean,
poisoned and poisoned_repair (soft R1) arms of the same source runs `sweep_f1.0_s{42,43,44}`, on the same test rows
(row sha256 asserted equal). The control numbers come from the source runs' final_test.json and are checked against
the committed experiments/results/04_sweep/final_test.csv. The FU2 summary.json and rounds.json in the test zip must
equal the validation copies committed under experiments/results/06_fu2/.

Writes to experiments/results/06_fu2/test/:
  fu2/cii/fu2_relabel_f1.0_s<seed>/final_test.json   copied unchanged
  test_recall.csv     per seed and arm: recall per class, macro recall, benign FAR, Syn test rows predicted BENIGN
  test_summary.csv    per arm: mean / min / max over the 3 seeds
  h3.csv              H3 (prereg_followup_fu2.md) per seed: Syn >= 0.95, LDAP and UDP >= 0.80, benign FAR <= 0.05

"syn_to_benign" (Syn test rows predicted BENIGN, the poison's direction) is a post-hoc descriptive number, not in
the pre-registration.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import load_json

SEEDS = (42, 43, 44)
CONTROLS = ("clean", "poisoned", "poisoned_repair")
CLASSES = ("BENIGN", "Syn", "LDAP", "UDP")
VAL = Path("experiments/results/06_fu2")


def row(seed, arm, t, decisions_commit=""):
    return {"seed": seed, "arm": arm, **{k: t["recall"].get(k, 0.0) for k in CLASSES},
            "macro_recall": t["macro_recall"], "benign_far": t["benign_far"], "n_benign": t["n_benign"],
            "syn_to_benign": t["confusion"]["Syn"].get("BENIGN", 0), "n_syn": t["support"]["Syn"],
            "test_rows": t["n_rows"], "test_rows_sha256": t["rows_sha256"], "decisions_commit": decisions_commit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fu2", required=True, help="fu2/ folder of fu2_test.zip (cii/fu2_relabel_f1.0_s<seed>/)")
    ap.add_argument("--src", required=True, help="nids_out of notebook 04 (runs/cii/sweep_f1.0_s<seed>/<arm>/)")
    ap.add_argument("--dest", default=str(VAL / "test"))
    a = ap.parse_args()
    fu2, src, dest = Path(a.fu2), Path(a.src), Path(a.dest)
    committed = pd.read_csv("experiments/results/04_sweep/final_test.csv")

    rows, h3 = [], []
    for seed in SEEDS:
        rel = Path("cii") / f"fu2_relabel_f1.0_s{seed}"
        for f in ("summary.json", "poisoned_relabel/rounds.json"):
            assert (fu2 / rel / f).read_bytes() == (VAL / "fu2" / rel / f).read_bytes(), f"{rel / f}: not the validated file"
        s, f = load_json(fu2 / rel / "summary.json"), load_json(fu2 / rel / "final_test.json")
        ft = f["test"]
        assert f["final_classes"] == s["final_classes"] and f["arm"] == s["arm"] == "poisoned_relabel"
        for arm in CONTROLS:
            ct = load_json(src / "runs/cii" / f"sweep_f1.0_s{seed}" / arm / "final_test.json")["test"]
            assert ct["rows_sha256"] == ft["rows_sha256"] and ct["support"] == ft["support"], f"s{seed} {arm}: different rows"
            c = committed[(committed["run"] == f"sweep_f1.0_s{seed}") & (committed["arm"] == arm)].iloc[0]
            assert c["test_rows_sha256"] == ct["rows_sha256"] and abs(c["benign_far"] - ct["benign_far"]) < 1e-12
            assert all(abs(c[f"recall_{k}"] - ct["recall"].get(k, 0.0)) < 1e-12 for k in CLASSES), "04_sweep table differs"
            rows.append(row(seed, arm, ct))
        rows.append(row(seed, "poisoned_relabel", ft, f["decisions_commit"]))
        r = ft["recall"]
        h3.append({"seed": seed, "Syn": r["Syn"], "LDAP": r["LDAP"], "UDP": r["UDP"], "benign_far": ft["benign_far"],
                   "met": r["Syn"] >= 0.95 and r["LDAP"] >= 0.80 and r["UDP"] >= 0.80 and ft["benign_far"] <= 0.05})
        (dest / "fu2" / rel).mkdir(parents=True, exist_ok=True)
        shutil.copy(fu2 / rel / "final_test.json", dest / "fu2" / rel / "final_test.json")

    df = pd.DataFrame(rows)
    df.to_csv(dest / "test_recall.csv", index=False)
    cols = list(CLASSES) + ["macro_recall", "benign_far", "syn_to_benign"]
    summ = df.groupby("arm", sort=False)[cols].agg(["mean", "min", "max"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.insert(0, "n_seeds", df.groupby("arm", sort=False).size())
    summ.reset_index().to_csv(dest / "test_summary.csv", index=False)
    pd.DataFrame(h3).to_csv(dest / "h3.csv", index=False)
    print(df[["seed", "arm", *cols]].round(4).to_string(index=False))
    print(pd.DataFrame(h3).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
