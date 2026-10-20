"""Live demo from SAVED outputs only: inference on CPU, nothing trained or written.

Three interactions (each prints a short table):
  python demo.py flow   --label Syn --index 0   # one protected TEST flow -> model input -> predictions
  python demo.py poison                         # the poisoned buffer and what the teacher thinks of it
  python demo.py gate   --arm poisoned          # every gate inequality, recomputed from saved metrics

Inputs are found like notebook 03 does (or pass --results / --prepared):
  --results : nids_out folder with runs/<scenario>/<run>/ (e.g. dvdhere/nids-results/nids_out)
  --prepared: nids_out folder with clean_*.parquet (results_01, inside the code dataset)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import s5_explain as X
from common import LABEL_COL, load_json

SHOW = ["flow duration", "total fwd packets", "fwd packet length max", "flow bytes/s", "syn flag count",
        "fwd header length"]


def flow(a, run, results, prepared):
    sc = load_json(results / a.scenario / "scaler.json")
    cm = load_json(results / "class_map.json")
    excl = set(load_json(results / "protected_exclusions.json")["excluded_rids"])
    d = pd.read_parquet(prepared / "clean_test.parquet")
    d = d[(d[LABEL_COL] == a.label) & ~d["_rid"].isin(excl)].sort_values("_rid")
    row = d.iloc[[a.index]]
    print(f"test flow rid {int(row['_rid'].iloc[0])}, true label {a.label}, source {row['_src'].iloc[0]}")
    feats = [f for f in SHOW if f in sc["features_numeric"]]
    print(X.worked_example(row, sc, feats)[["feature", "raw x", "z", "float32 input"]].to_string(index=False))
    print("protocol:", X.protocol_example(row, sc)["raw protocol"], "-> id",
          X.protocol_example(row, sc)["integer id fed to nn.Embedding"])
    x, c, _ = X._rows(prepared, "test", row["_rid"].to_numpy(), sc, cm)
    last = f"A{len(run['config']['schedule']) - 1}"
    models = {"T (frozen teacher)": run["dir"] / "T.pt",
              **{f"{arm} final ({last})": run["dir"] / arm / f"{last}.pt" for arm in X.ARMS}}
    rows = []
    for name, path in models.items():
        if path.exists():
            m, cls = X.model_from_ckpt(path, run["config"]["args"])
            p = X._proba(m, x, c)[0]
            top = np.argsort(-p)[:3]
            rows.append({"model": name, "prediction": cls[top[0]],
                         "top-3": ", ".join(f"{cls[i]} {p[i]:.3f}" for i in top)})
    print(pd.DataFrame(rows).to_string(index=False))


def poison(a, run, results, prepared):
    buf = run["buffers"]["poisoned"][1]
    print("poisoned buffer before E1 (stored label vs true label):")
    print(pd.crosstab(buf["true_label"], buf["label"]).to_string())
    probs = X.buffer_probabilities(results, prepared, run, a.scenario)
    print("\nmean probabilities per group (T's p(stored) = the weight R1 WOULD give; diagnostic):")
    cols = ["model", "group", "n", "mean p(stored label)", "mean p(true label)", "share predicted = true"]
    print(X.poison_diagnostics(probs)[cols].to_string(index=False))


def gate(a, run, results, prepared):
    g = X.gate_check(run["rounds"][a.arm], run["config"]["args"])
    print(f"arm {a.arm}: every gate inequality, recomputed from rounds.json (matches the saved decision)")
    print(g.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["flow", "poison", "gate"])
    ap.add_argument("--results", default=None)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--scenario", default="cii")
    ap.add_argument("--run", default="main")
    ap.add_argument("--label", default="Syn")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--arm", default="poisoned", choices=list(X.ARMS))
    a = ap.parse_args()
    results = Path(a.results) if a.results else X.find_dir(f"runs/{a.scenario}/{a.run}/config.json", "NIDS_RESULTS")
    prepared = Path(a.prepared) if a.prepared else X.find_dir("clean_train.parquet", "NIDS_PREPARED")
    assert results, "results folder not found: pass --results"
    if a.what != "gate":
        assert prepared, "prepared parquet folder not found: pass --prepared"
    run = X.load_run(results, a.scenario, a.run)
    pd.set_option("display.width", 200)
    {"flow": flow, "poison": poison, "gate": gate}[a.what](a, run, results, prepared)


if __name__ == "__main__":
    main()
