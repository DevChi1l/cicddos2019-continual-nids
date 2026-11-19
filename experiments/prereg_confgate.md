# Pre-registration CG: a confidence-aware gate (POST HOC follow-up, 2026-11-19)

Committed before any CG run. Code: `s15_confgate.py` (this commit). Kaggle: `kaggle/cg_val/` then `kaggle/cg_test/`.

## Why (post hoc)
Run `main` (four classes, f = 0.5) and RF (refresh, 10 % more flips per update) showed the same limitation: the recall
gate accepts a poisoned update whose argmax is unchanged but whose confidence falls (main: mean p(Syn) 0.994 -> ~0.79;
RF test: 0.7707 vs 0.9922 clean). Repair therefore never ran. CG is designed **after** seeing those results, so every
CG result is labelled post hoc. Nothing in the original claims changes.

## The one change
s3's existing option `--gate-conf-drop 0.02`: a candidate is rejected if, for any old attack class k,
mean p(k | protected val rows of k) < (same quantity for the last accepted model) - 0.02. All other gate terms,
recipe, buffer, poisoning and the soft R1 repair are unchanged.

**Threshold 0.02, fixed now from committed clean logs only:** the largest one-update fall in an old class's mean
confidence in any clean arm on record is 0.0074 (Syn, RF clean, 36 updates), 0.0074 (Syn, sweep clean, 26 updates),
0.0004 (TFTP, RF clean); 0.0108 (LDAP at E2, FU2 relabel arm, seeds 42-44). 0.02 is about 2x the largest.

## Settings (validation decides; test scored once, after this file's results are committed)
- **CG-RF:** tag ref6 (E0 BENIGN + Syn + TFTP, then 6 TFTP refresh updates), rho 0.10 and 0.05, seeds 42-44,
  `--poison-every-update`, arms clean / poisoned / poisoned_repair.
- **CG-MAIN:** four-class CII (BENIGN + Syn, + LDAP, + UDP), f = 0.5, seeds 42-44, same arms.

## Hypotheses (registered predictions; reported whatever happens)
- **CG0 (descriptive):** each clean arm makes the same decisions as the matching committed run (RF clean; main clean
  for seed 42). Not a pass/fail gate: the Kaggle software image may have changed since October.
- **CG1 (no false alarms of the gate):** every clean-arm update is accepted (36 RF updates, 6 four-class updates).
  If any clean update is rejected, the threshold is too strict for that setting and **no CG claim is made there.**
- **CG2 (the gate now sees the poison):** the poisoned C1 at the first update is rejected in 3/3 seeds at rho 0.10
  and in 3/3 seeds of CG-MAIN.
- **CG3 (repair restores confidence):** in poisoned_repair, the deployed final model's mean p(Syn) on held-out test is
  >= 0.95 in every run (RF without the confidence gate: 0.7707 at rho 0.10, 0.9027 at rho 0.05).
- **CG4 (still learning):** poisoned_repair accepts >= 5 of 6 updates in every CG-RF run, and learns all four classes
  in 3/3 CG-MAIN seeds.

## Stopping and reporting
- One execution per setting; no threshold search, no re-runs with other values. If a run crashes, it is resumed
  (deterministic) and the crash is recorded.
- Results go to `experiments/results/11_confgate/` (summary.json committed before test; test.json after).
- Scope: one attacked class, 3 seeds, no significance claim. A confidence gate can also be attacked (an adversary
  who keeps confidence high); not tested.
