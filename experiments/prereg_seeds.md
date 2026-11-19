# Pre-registration SEEDS: five more seeds for the main defended results (2026-11-19)

Committed before any SEEDS run. Code: `s17_seeds.py`. Kaggle: `kaggle/seeds_val/` then `kaggle/seeds_test/`.

## Why
The main results rest on 3 seeds (42-44). SEEDS repeats the same settings with seeds 45-49 (fixed now). Nothing is
tuned: same code paths, recipe, gate (0.05 / 0.80 / 0.05 / 0.05, no confidence term), buffer and attack.

## Status of each setting (unchanged from the original)
- 4-class soft repair at f = 0.75 / 0.9 / 0.95: the sweep's REGISTERED method; new seeds are a registered replication.
- 4-class relabel repair at f = 1.0: FU2's method, which is POST HOC (chosen after the sweep); new seeds stay post hoc.
- 5-class (FIVE schedule) soft at f = 0.9 and relabel at f = 1.0: FIVE is POST HOC; new seeds stay post hoc.
  FIVE's clean pilot (seeds 42-44) already passed; no new pilot. The clean arm in every run is the control.

## Predictions (reported whatever happens; counts over the 5 NEW seeds, then pooled with 42-44 as descriptive)
- **S1:** 4-class, f in {0.75, 0.9, 0.95}: poisoned_repair (soft) learns all 4 classes in 15/15 runs; gate only in
  fewer than 15/15.
- **S2:** 4-class, f = 1.0: relabel repair learns all 4 classes in 5/5; gate only in 0/5.
- **S3:** 5-class: soft at f = 0.9 learns all 5 in 5/5; relabel at f = 1.0 learns all 5 in 5/5; gate only < 5/5 and 0/5.
- **S4:** benign FAR <= 0.005 for every repaired deployed model; every clean arm learns all its classes.
"Learns a class" = held-out test recall > 0 for that class in the deployed model (as in the sweep).

## Rules
One execution, no re-runs with other settings; a crash is resumed (deterministic) and recorded. Validation decides;
`experiments/results/13_seeds/summary.json` is committed before test, which is scored once. No significance test is
registered; counts and ranges are reported.
