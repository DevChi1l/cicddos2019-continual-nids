"""Live demo from SAVED outputs only: inference on CPU, nothing trained or written.

Three interactions on the `main` run's saved outputs (each prints a short table):
  python demo.py flow   --label Syn --index 0   # one protected TEST flow -> model input -> predictions
  python demo.py poison                         # the poisoned buffer and what the teacher thinks of it
  python demo.py gate   --arm poisoned          # every gate inequality, recomputed from saved metrics

Two more that need ONLY the files committed in git (no data, no checkpoints, no GPU; seconds on a laptop):
  python demo.py fu2 --seed 42   # PRIMARY RESULT: f = 1.0 attack, C1 rejected, relabel R1 accepted, test vs controls
  python demo.py rf              # LIMITATION (post hoc RF): confidence falls, recall and the gate do not react
  python demo.py walkthrough               # guided, deterministic ~3-minute FU2 story (five timed steps)
  python demo.py walkthrough --story rf    # the same format for the RF confidence-loss limitation
  python demo.py walkthrough --pause       # wait for Enter between steps (for a live presentation)
Every number printed by fu2 / rf / walkthrough is read from a saved file; nothing is simulated or re-predicted.

Inputs are found like notebook 03 does (or pass --results / --prepared):
  --results : nids_out folder with runs/<scenario>/<run>/ (e.g. dvdhere/nids-results/nids_out)
  --prepared: nids_out folder with clean_*.parquet (results_01, inside the code dataset)
"""
from __future__ import annotations

import argparse
import sys
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


# ---- guided walkthrough (committed files only; deterministic) ---------------------------------------
def _step(a, n, total, clock, title):
    if a.pause and n > 1 and sys.stdin.isatty():
        input("\n[Enter] next step ")
    print(f"\n{'=' * 100}\nSTEP {n}/{total}  [{clock}]  {title}\n{'=' * 100}")


def _say(text):
    print(f"  SAY: {text}")


def walkthrough_fu2(a):
    runs = {s: COMMITTED / "06_fu2" / "fu2" / "cii" / f"fu2_relabel_f1.0_s{s}" for s in (42, 43, 44)}
    summ = {s: load_json(r / "summary.json") for s, r in runs.items()}
    rounds = {s: load_json(r / "poisoned_relabel" / "rounds.json") for s, r in runs.items()}
    dec = pd.read_csv(COMMITTED / "04_sweep" / "decisions.csv")
    soft = dec[(dec["poison_frac"] == 1.0) & (dec["arm"] == "poisoned_repair")].sort_values(["seed", "experience", "attempt"])
    ts = pd.read_csv(COMMITTED / "06_fu2" / "test" / "test_summary.csv").set_index("arm")
    total = 5

    _step(a, 1, total, "0:00-0:30", "Setting and attack (four labels, CII, seeds 42-44)")
    print("  E0 = BENIGN + Syn  ->  E1 + LDAP  ->  E2 + UDP; replay buffer 2,000 rows per class per accepted experience.")
    for s in (42, 43, 44):
        print(f"  seed {s}: stored Syn rows flipped to BENIGN before E1 = {rounds[s][0]['n_replay_flipped']} of 2,000 "
              f"(f = {summ[s]['poison_frac']}); frozen teacher unchanged = {summ[s]['T_unchanged']} "
              f"(sha256 {summ[s]['T_sha256'][:12]})")
    _say("The attacker can only edit stored labels. Here it flipped every stored Syn label. The teacher T is the E0 "
         "model, frozen and hash-checked.")

    _step(a, 2, total, "0:30-1:15", "Plain update C1 and the REGISTERED soft repair both fail (saved gate decisions)")
    rows = []
    for s in (42, 43, 44):
        for r in rounds[s]:
            if r["attempt"] == "C1":
                v, ref = r["val"], r["ref_val"]
                rows.append({"seed": s, "update": f"E{r['experience']}", "attempt": "C1 (stored labels)",
                             "Syn val recall": round(v["recall"]["Syn"], 4),
                             "floor": round(ref["recall"]["Syn"] - GATE["gate_old_drop"], 4),
                             "decision": "ACCEPT" if r["accepted"] else "REJECT", "saved reason": "; ".join(r["gate_reasons"])})
    for _, r in soft.iterrows():
        if r["attempt"] == "R1":
            rows.append({"seed": int(r["seed"]), "update": f"E{int(r['experience'])}", "attempt": "soft R1 (sweep)",
                         "Syn val recall": round(r["Syn_val_recall"], 4), "floor": round(r["Syn_floor"], 4),
                         "decision": "ACCEPT" if r["accepted"] else "REJECT",
                         "saved reason": f"teacher weight: flipped rows {r['w_flipped']:.4f}, clean rows {r['w_clean']:.4f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    _say("The gate catches it: Syn recall falls to 0 against a floor of about 0.945. Soft repair down-weights the "
         "flipped rows to about 0.003, but no correct Syn label is left, so it also fails. The gate then keeps the old "
         "model, which never learns LDAP or UDP.")

    _step(a, 3, total, "1:15-2:00", "Teacher relabelling R1 (FU2, post hoc method): what T did, and the gate's verdict")
    rows = []
    for s in (42, 43, 44):
        for r in rounds[s]:
            if r["attempt"] == "R1":
                tw, v, ref = r["teacher_weights"], r["val"], r["ref_val"]
                rows.append({"seed": s, "update": f"E{r['experience']}", "rows relabelled": tw["n_relabelled"],
                             "to their true class": tw["n_relabelled_to_true"], "of them attacked": tw["n_relabelled_flipped"],
                             "quarantined (weight 0)": tw["n_quarantined"],
                             "Syn val recall": round(v["recall"]["Syn"], 4),
                             "floor": round(ref["recall"]["Syn"] - GATE["gate_old_drop"], 4),
                             "decision": "ACCEPT" if r["accepted"] else "REJECT"})
    print(pd.DataFrame(rows).to_string(index=False))
    _say("Where T is at least 90% sure that a stored BENIGN row is Syn, the row is trained as Syn: 1,994-1,995 rows per "
         "repair, all to their true class. The gate accepts at E1 and E2 in every seed. E1 replays the earlier "
         "diagnostic, so the new evidence is E2 and the test.")

    _step(a, 4, total, "2:00-2:40", "Held-out test, mean over seeds 42-44, the same 46,527 rows for every arm")
    print("  relabel: scored once after FU2's validation commit. clean / gate only / soft repair: the sweep's f = 1.0 "
          "controls\n  (test computed inside those runs, never used for a decision; sweep deviation D1).")
    names = {"clean": "clean (no poison)", "poisoned": "gate only", "poisoned_repair": "soft repair (registered)",
             "poisoned_relabel": "teacher relabel (FU2)"}
    t = pd.DataFrame({names[k]: {m: round(ts.loc[k, f"{m}_mean"], 4) for m in ("Syn", "LDAP", "UDP", "benign_far")}
                      for k in names}).T
    print(t.to_string())
    _say("Relabelling learns all four classes, within about 0.01 of clean, at the same benign false-alarm rate. Gate "
         "only and soft repair score 0 on LDAP and UDP: the stale model calls them benign.")

    _step(a, 5, total, "2:40-3:00", "Scope: what we claim and what we do not")
    print("  - Claimed: a four-label continual defence against 100% targeted replay-label flipping of one class (Syn).\n"
          "    CII only, 3 seeds, no significance test. Relabelling is POST HOC: chosen after the sweep's diagnostic.\n"
          "  - Not claimed: a 12- or 14-label defence (unsolved), other attacked classes, CI, full-data training.\n"
          "  - Known limitation: the recall-based gate misses confidence loss (python demo.py walkthrough --story rf).")


def walkthrough_rf(a):
    tv = pd.read_csv(COMMITTED / "10_rf" / "rf" / "trajectory_val.csv")
    dg = pd.read_csv(COMMITTED / "10_rf" / "rf" / "diag_val.csv")
    ts = pd.read_csv(COMMITTED / "10_rf" / "test" / "test_summary.csv")
    total = 4
    _step(a, 1, total, "0:00-0:30", "Setting (RF, post hoc): fixed classes, traffic refresh, no new class")
    print("  E0 = BENIGN + Syn + TFTP; six updates of fresh BENIGN + TFTP rows; Syn exists only in the replay buffer.\n"
          "  Before every update the attacker flips 200 more stored Syn labels (10% of 2,000): 60% after E6.")
    _step(a, 2, total, "0:30-1:30", "Saved gate decisions for every poisoned update (rho 0.10, seeds 42-44)")
    po = tv[(tv["rho"] == 0.10) & (tv["arm"] == "poisoned") & (tv["t"] > 0)]
    piv = po.pivot_table(index="t", columns="seed", values=["flipped_share", "c1_accepted", "c1_Syn_margin", "Syn_mean_p"],
                         aggfunc="first")
    print(piv.round(4).to_string())
    rep = tv[(tv["rho"] == 0.10) & (tv["arm"] == "poisoned_repair") & (tv["t"] > 0)]
    print(f"  poisoned arm: C1 accepted {int(po['c1_accepted'].sum())}/{len(po)}; repair arm: R1 ran {int(rep['r1_ran'].sum())} times")
    _say(f"Every update passes; the smallest Syn gate margin is {po['c1_Syn_margin'].min():.4f}, so the repair is never "
         "asked for. Meanwhile mean p(Syn) keeps falling (it ticks up once, at t = 4).")
    _step(a, 3, total, "1:30-2:15", "What the teacher WOULD have done (forced R1 diagnostic: logged, never deployed)")
    d = dg[dg["rho"] == 0.10].groupby("t").agg(would_pass=("would_pass_gate", "mean"), Syn_mean_p=("Syn_mean_p", "mean"),
                                              w_flipped=("mean_w_flipped", "mean"), w_clean=("mean_w_clean_known", "mean"))
    print(d.round(4).to_string())
    _say("The diagnostic would have passed and kept confidence near 0.99, but it was never deployed and never scored on "
         "test. We do not claim that RF repaired anything.")
    _step(a, 4, total, "2:15-3:00", "Held-out test after E6 (mean over seeds; scored once after the validation commit)")
    t = ts[(ts["model"] == "A6") & (ts["arm"] != "poisoned_repair")]
    print(t[["rho", "arm", "flipped_share", "Syn_recall_mean", "Syn_mean_p_mean", "benign_far_mean"]].round(4).to_string(index=False))
    _say("Recall 0.9943 against 0.9952 clean, but confidence 0.7707 against 0.9922. Three fixed classes, traffic refresh, "
         "not new-class arrival: a limitation of a recall-only gate, not a defence.")


def walkthrough(a):
    print(f"Guided walkthrough: {'FU2 four-class defence (PRIMARY result)' if a.story == 'fu2' else 'RF confidence-loss limitation'}"
          " - every value is read from experiments/results/ (committed); nothing is re-predicted.")
    (walkthrough_fu2 if a.story == "fu2" else walkthrough_rf)(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["flow", "poison", "gate", "fu2", "rf", "walkthrough"])
    ap.add_argument("--story", default="fu2", choices=["fu2", "rf"], help="walkthrough only")
    ap.add_argument("--pause", action="store_true", help="walkthrough only: wait for Enter between steps")
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
    if a.what in ("fu2", "rf", "walkthrough"):
        return {"fu2": fu2, "rf": rf, "walkthrough": walkthrough}[a.what](a)
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
