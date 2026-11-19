# Pre-registration FIVE: the four-class defence with a fifth class (POST HOC follow-up, 2026-11-19)

Committed before any FIVE run. Code: `s16_five.py`. Kaggle: `kaggle/five_val/` then `kaggle/five_test/`.

## Why (post hoc)
The defended result covers four classes. The 12-label schedule failed even without poison, so it says nothing about
whether the defence works with more than four classes. FIVE adds one class whose clean behaviour is already known to
be easy: TFTP (test recall 0.9964 in the clean 12-label run, 0.9957 in the joint oracle; clean RF updates with TFTP
were all accepted). It was chosen after seeing those results, so every FIVE result is post hoc.

## Design
- Tag `five`, CII: E0 = BENIGN + Syn, E1 + LDAP, E2 + UDP, E3 + TFTP. Attack cap 13,890 (unchanged rule: at most
  one CII benign portion and at most the smallest class; LDAP has 13,890).
- Recipe, gate (0.05 / 0.80 / 0.05 / 0.05, no confidence term), buffer (2,000 per class) and attack (stored Syn ->
  BENIGN, once before E1) exactly as in the sweep and FU2.
- Seeds 42-44. f = 0.9 with the registered soft R1; f = 1.0 with the FU2 relabel R1 (threeway+relabel, thresholds
  0.9 / 0.1). Arms clean / poisoned (gate only) / poisoned_repair.

## Stopping rule (decided now)
Clean pilot first (clean arm, seeds 42-44, validation). **Proceed only if every clean update is accepted in all three
seeds.** Otherwise stop, report that the fifth class breaks the clean protocol, and run nothing else. No other class,
schedule or threshold is tried.

## Hypotheses (reported whatever happens)
- **F1:** every clean update is accepted (3 updates x 3 seeds); the clean deployed model learns all 5 classes in 3/3.
- **F2:** at f = 0.9, soft repair deploys a model that learns all 5 classes in 3/3 seeds; gate only does so in fewer
  than 3/3.
- **F3:** at f = 1.0, relabel repair learns all 5 classes in 3/3 seeds; gate only in 0/3.
- **F4:** benign FAR of every repaired deployed model <= 0.005 on held-out test.

Validation decides; `experiments/results/12_five/summary.json` is committed before test, which is scored once.
Scope: one attacked class, 3 seeds, no significance claim.
