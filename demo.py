"""Live demo from SAVED outputs only: inference on CPU, nothing trained or written.

Three interactions on the `main` run's saved outputs (each prints a short table):
  python demo.py flow   --label Syn --index 0   # one protected TEST flow -> model input -> predictions
  python demo.py poison                         # the poisoned buffer and what the teacher thinks of it
  python demo.py gate   --arm poisoned          # every gate inequality, recomputed from saved metrics

Two more that need ONLY the files committed in git (no data, no checkpoints, no GPU; seconds on a laptop):
  python demo.py fu2 --seed 42   # PRIMARY RESULT: f = 1.0 attack, C1 rejected, relabel R1 accepted, test vs controls
  python demo.py rf              # LIMITATION (post hoc RF): confidence falls, recall and the gate do not react

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


# ---- committed-evidence demos (experiments/results/, read only) -----------------------------------
COMMITTED = Path(__file__).resolve().parent / "experiments" / "results"
GATE = {"gate_old_drop": 0.05, "gate_new_min": 0.80, "gate_far_max": 0.05, "gate_newfar_max": 0.05,
        "gate_conf_drop": None}  # the registered gate (s3 defaults) of every run shown here


def fu2(a):
    run = COMMITTED / "06_fu2" / "fu2" / "cii" / f"fu2_relabel_f1.0_s{a.seed}"
    summ = load_json(run / "summary.json")
    rounds = load_json(run / "poisoned_relabel" / "rounds.json")
    print(f"FU2, seed {a.seed}: four labels (E0 BENIGN+Syn, E1 +LDAP, E2 +UDP), CII. Before E1 the attacker flipped "
          f"{rounds[0]['n_replay_flipped']} of the 2,000 stored Syn labels to BENIGN (f = {summ['poison_frac']}); flips "
          f"persist. R1 = {summ['r1_method']} (post hoc method). Teacher T unchanged: {summ['T_unchanged']}.\n")
    g = X.gate_check(rounds, GATE)  # asserts the recomputed decision equals the saved one
    g["value"], g["bound"], g["margin"] = g["value"].round(4), g["bound"].round(4), g["margin"].round(4)
    print("1) Every gate inequality on validation, recomputed from rounds.json (matches the saved decisions):")
    print(g.to_string(index=False))
    print("\n2) What the relabel R1 did to the replay buffer (T's prediction replaces the stored label):")
    for r in rounds:
        tw = r.get("teacher_weights")
        if tw:
            print(f"   E{r['experience']} R1: relabelled {tw['n_relabelled']} rows, {tw['n_relabelled_to_true']} to their "
                  f"true class; {tw['n_relabelled_flipped']} of them were attacked rows -> accepted: {r['accepted']}")
    tr = pd.read_csv(COMMITTED / "06_fu2" / "test" / "test_recall.csv")
    t = tr[tr["seed"] == a.seed].set_index("arm")[["Syn", "LDAP", "UDP", "macro_recall", "benign_far"]].round(4)
    print(f"\n3) Held-out test, seed {a.seed} ({int(tr['test_rows'].iloc[0]):,} rows, the same rows for every arm). "
          f"poisoned_relabel was scored once after FU2's validation commit; clean / poisoned / poisoned_repair are the "
          f"sweep's f = 1.0 controls:")
    print(t.to_string())
    m = tr.groupby("arm")[["Syn", "LDAP", "UDP", "benign_far"]].mean().round(4)
    print("\n   mean over seeds 42-44:")
    print(m.to_string())


def rf(a):
    tv = pd.read_csv(COMMITTED / "10_rf" / "rf" / "trajectory_val.csv")
    v = tv[(tv["rho"] == 0.10) & (tv["arm"].isin(["clean", "poisoned"]))]
    piv = v.pivot_table(index="t", columns="arm", values=["Syn_recall", "Syn_mean_p"], aggfunc="mean").round(4)
    acc = v[v["t"] > 0].groupby("arm")["c1_accepted"].agg(lambda s: f"{int(s.sum())}/{len(s)}")
    print("RF (post hoc, fixed-class traffic refresh: E0 BENIGN+Syn+TFTP, six updates of fresh BENIGN+TFTP, no new "
          "class). rho = 0.10: 200 more stored Syn labels flipped before every update (60% after E6).\n")
    print("Validation trajectory, mean over seeds 42-44 (t = update):")
    print(piv.to_string())
    print(f"C1 accepted: {acc.to_dict()}; R1 ran: {int(v['r1_ran'].sum())} times")
    ts = pd.read_csv(COMMITTED / "10_rf" / "test" / "test_summary.csv")
    t = ts[(ts["model"] == "A6") & (ts["arm"] != "poisoned_repair")]
    print("\nHeld-out test after E6 (mean over seeds; poisoned_repair is identical to poisoned):")
    print(t[["rho", "arm", "flipped_share", "Syn_recall_mean", "Syn_mean_p_mean", "benign_far_mean"]].round(4)
          .to_string(index=False))
    print("\nReading: recall held and no update was rejected, so no repair was deployed; the confidence loss stayed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["flow", "poison", "gate", "fu2", "rf"])
    ap.add_argument("--seed", type=int, default=42, choices=[42, 43, 44], help="fu2 only")
    ap.add_argument("--results", default=None)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--scenario", default="cii")
    ap.add_argument("--run", default="main")
    ap.add_argument("--label", default="Syn")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--arm", default="poisoned", choices=list(X.ARMS))
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.what in ("fu2", "rf"):
        return {"fu2": fu2, "rf": rf}[a.what](a)
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
