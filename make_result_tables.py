"""Small result tables for git, from SAVED run outputs only (trains and scores nothing).

  python make_result_tables.py --out <nids_out of notebook 04> --dest experiments/results

Writes CSV/JSON only (no checkpoints, no parquet):
  04_sweep/final_test.csv, test_summary.csv, test_syn.png   copies of `s6_sweep.py --test` output
  04_sweep/deployed_models.csv     per run and arm: what each update did and the DEPLOYED model's test metrics
  04b_all12/decisions.csv          every gate decision (val), worst old class vs its floor, min new recall
  04b_all12/diagnostics.csv        forced R1 (DIAGNOSTIC, never a decision)
  04b_all12/final_test.csv         deployed model of each arm on held-out test
  04b_all12/rejected_confusion.csv for every rejected attempt: where the failing class's val rows went
  04b_all12/schedule.json, run_configs.json

"Deployed" = the last accepted model (s3 keeps the previous model when no candidate passes). The fallback
check compares weight hashes: A_t must equal the accepted candidate, or A_{t-1} when every attempt failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from common import load_json

ARMS = ("clean", "poisoned", "poisoned_repair")


def weights(path: Path) -> str:
    sd = torch.load(path, map_location="cpu")["state"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


def arm_trace(rdir: Path, arm: str) -> tuple[list[str], bool]:
    """Per update 'E1 C1' / 'E1 R1' (accepted) or 'E1 kept' (all attempts rejected), and the fallback check."""
    rounds = load_json(rdir / arm / "rounds.json")
    prev, steps, ok = weights(rdir / "A0.pt"), [], True
    for t in sorted({r["experience"] for r in rounds}):
        acc = [r for r in rounds if r["experience"] == t and r["accepted"]]
        now = weights(rdir / arm / f"A{t}.pt")
        if acc:
            ok &= now == weights(rdir / arm / f"E{t}_{acc[0]['attempt']}.pt")
            steps.append(f"E{t} {acc[0]['attempt']}")
        else:
            ok &= now == prev
            steps.append(f"E{t} kept")
        prev = now
    return steps, ok


def deployed(rdir: Path) -> list[dict]:
    cfg = load_json(rdir / "config.json")
    rows = []
    for arm in ARMS:
        rounds = load_json(rdir / arm / "rounds.json")
        steps, ok = arm_trace(rdir, arm)
        ft = load_json(rdir / arm / "final_test.json")
        t = ft["test"]
        r1 = [r for r in rounds if r["attempt"] == "R1"]
        rows.append({"run": rdir.name, "scenario": rdir.parent.name, "poison_frac": cfg["args"]["poison_frac"],
                     "seed": cfg.get("seed", 42), "arm": arm, "updates": ", ".join(steps),
                     "updates_accepted": sum("kept" not in s for s in steps),
                     "updates_kept_previous": sum("kept" in s for s in steps),
                     "r1_ran": len(r1), "r1_accepted": sum(r["accepted"] for r in r1),
                     "fallback_ok": ok, "classes_learned": " ".join(ft["final_classes"]),
                     "test_benign_far": t["benign_far"], "test_macro_recall": t["macro_recall"],
                     "test_rows": t["n_rows"], "test_rows_sha256": t["rows_sha256"],
                     **{f"test_recall_{k}": v for k, v in t["recall"].items()}})
    return rows


def decisions(rdir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    args = load_json(rdir / "config.json")["args"]
    drop, tgt = args["gate_old_drop"], args["poison_target"]
    dec, conf, diag = [], [], []
    for arm in ARMS:
        recs = [(r, "decision") for r in load_json(rdir / arm / "rounds.json")]
        if (rdir / arm / "r1_diag.json").exists():
            recs += [(r, "diagnostic") for r in load_json(rdir / arm / "r1_diag.json")]
        for r, kind in recs:
            v, ref = r["val"], r["ref_val"]
            # old classes = the reference model's classes (r1_diag's "old_classes" is logged after acceptance)
            old = {k: (v["recall"].get(k, 0.0), ref["recall"][k] - drop) for k in ref["recall"] if k != "BENIGN"}
            worst = min(old, key=lambda k: old[k][0] - old[k][1])
            new_min = min(r["new_classes"], key=lambda k: v["recall"].get(k, 0.0))
            row = {"scenario": rdir.parent.name, "arm": arm, "experience": r["experience"],
                   "new_classes": " ".join(r["new_classes"]),
                   f"{tgt}_val_recall": v["recall"].get(tgt), f"{tgt}_floor": ref["recall"][tgt] - drop,
                   "worst_old_class": worst, "worst_old_recall": old[worst][0], "worst_old_floor": old[worst][1],
                   "min_new_class": new_min, "min_new_recall": v["recall"].get(new_min, 0.0),
                   "benign_far": v["benign_far"], "new_class_far": v["new_class_far"],
                   "gate_reasons": "; ".join(r["gate_reasons"]) or "-"}
            if kind == "diagnostic":
                diag.append({**row, "method": r["method"], "label": "DIAGNOSTIC (not a decision)",
                             "would_pass_gate": r["would_pass_gate"]})
                continue
            dec.append({**row, "attempt": r["attempt"], "accepted": r["accepted"]})
            if not r["accepted"]:  # where did the failing classes' val rows go?
                failing = [k for k in old if old[k][0] < old[k][1]] + \
                          [k for k in r["new_classes"] if v["recall"].get(k, 0.0) < args["gate_new_min"]]
                for k in failing:
                    row_c = v["confusion"][k]
                    n = sum(row_c.values())
                    top = sorted(row_c.items(), key=lambda kv: -kv[1])[:3]
                    conf.append({"scenario": rdir.parent.name, "arm": arm, "experience": r["experience"],
                                 "attempt": r["attempt"], "class": k, "val_rows": n,
                                 "recall": v["recall"].get(k, 0.0),
                                 "predicted_as (top 3)": "; ".join(f"{p} {c} ({c / n:.1%})" for p, c in top)})
    return dec, diag, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="nids_out folder of notebook 04 (runs/, sweep/, all12_schedule.json)")
    ap.add_argument("--dest", default="experiments/results")
    a = ap.parse_args()
    out, dest = Path(a.out), Path(a.dest)

    sw = dest / "04_sweep"
    sw.mkdir(parents=True, exist_ok=True)
    for f in ("final_test.csv", "test_summary.csv", "test_syn.png"):
        shutil.copy(out / "sweep" / f, sw / f)
    runs = sorted((out / "runs" / "cii").glob("sweep_*")) + [out / "runs" / "cii" / "repro_s42"]
    dep = pd.DataFrame([row for r in runs for row in deployed(r)])
    dep.to_csv(sw / "deployed_models.csv", index=False)

    pb = dest / "04b_all12"
    pb.mkdir(parents=True, exist_ok=True)
    al = sorted(out.glob("runs/*/all12_*"), key=lambda p: p.parent.name, reverse=True)  # cii, then ci
    dec, diag, conf = [], [], []
    for r in al:
        d1, d2, d3 = decisions(r)
        dec += d1
        diag += d2
        conf += d3
    fin = pd.DataFrame([row for r in al for row in deployed(r)])
    pd.DataFrame(dec).to_csv(pb / "decisions.csv", index=False)
    pd.DataFrame(diag).to_csv(pb / "diagnostics.csv", index=False)
    pd.DataFrame(conf).to_csv(pb / "rejected_confusion.csv", index=False)
    fin.to_csv(pb / "final_test.csv", index=False)
    shutil.copy(out / "all12_schedule.json", pb / "schedule.json")
    (pb / "run_configs.json").write_text(json.dumps(
        {r.parent.name: {k: load_json(r / "config.json")[k] for k in ("args", "seed", "schedule", "inputs",
                                                                       "protected_excluded_rows", "val_rows",
                                                                       "test_rows", "device", "torch")}
         for r in al}, indent=1))

    assert dep["fallback_ok"].all() and fin["fallback_ok"].all(), "a deployed model is not the expected one"
    assert fin["test_rows_sha256"].nunique() == 1 and dep["test_rows_sha256"].nunique() == 1
    print(f"{len(dep)} sweep arms, {len(fin)} all12 arms: fallback check OK, same test rows within each study")
    print(f"written to {sw} and {pb}")


if __name__ == "__main__":
    main()
