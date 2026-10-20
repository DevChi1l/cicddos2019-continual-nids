# Pre-registration: poison-fraction sweep (CII, CICDDoS2019)

Written 2026-10-21, **before any sweep run**. The git commit that adds this file is its timestamp.
Any change after that commit goes under "Deviations" at the end, with a date and a reason.

## What is fixed

| Item | Value |
|---|---|
| Data | Stage 1/1b outputs from `results_01` (the same files as run `main`; s3 checks their sha256) |
| Input sha256 (first 16 hex digits) | split_manifest bb1f5acbd8ca2cbc, class_map 24925278bb16fde4, clean_train 379e005649890b48, clean_val b47c2469fa0b3198, clean_test 1019948deae58350, protected_exclusions 2894d5c2e743e8e7, cii/experience_rids 5f7e4c325aebd92b, cii/scaler 9ec6876fa51c5c7e, cii/experience_manifest ae9f5bda0ec6c7a2 |
| Scenario / schedule | CII: E0 = BENIGN + Syn, E1 = BENIGN + LDAP, E2 = BENIGN + UDP (13,890 attack rows each; benign 19,055 / 19,054 / 19,054) |
| Model / training | s2 model, d 32, 3 layers, 4 heads, dropout 0.1; epochs 10 (E0) / 5 (update); batch 512; lr 1e-3 |
| Buffer | 2,000 rows per class per accepted experience |
| Poison | target Syn -> BENIGN, applied once before E1, flips persist (`poison()` as in `main`) |
| Gate (val only) | old attack recall >= ref - 0.05; new recall >= 0.80; benign FAR <= 0.05; new-class FAR <= 0.05; ref = last accepted model |
| Confidence | mean p(k \| val rows of k) logged for every old class; **not** part of the gate (`--gate-conf-drop` unset) |
| Decision repair (arm `poisoned_repair`) | R1 = soft rule, weight p_T(stored label) (as in `main`) |
| Diagnostic repairs (`--forced-r1`, arms `clean` and `poisoned`) | soft; threeway (p_T >= 0.9 keep, <= 0.1 weight 0, otherwise weight 0.5); threeway+relabel (p_T(stored) <= 0.1 and max p_T >= 0.9: label := T's prediction). Logged in `r1_diag.json`, **never used for a decision** |

Disclosure: the 0.9 / 0.1 thresholds were chosen after seeing `main`'s diagnostic (T gave flipped rows
mean p 0.004 and clean rows about 0.998). They are fixed from now on.

## Grid and run names

| Run name | poison_frac | seed | flags |
|---|---|---|---|
| `repro_s42` | 0.5 | 42 | defaults only (must reproduce `main`) |
| `sweep_f0.75_s42`, `_s43`, `_s44` | 0.75 | 42 / 43 / 44 | `--forced-r1 soft,threeway,threeway+relabel` |
| `sweep_f0.9_s{42,43,44}` | 0.9 | 42 / 43 / 44 | same |
| `sweep_f0.95_s{42,43,44}` | 0.95 | 42 / 43 / 44 | same |
| `sweep_f1.0_s{42,43,44}` | 1.0 | 42 / 43 / 44 | same |

`--seed` changes model initialisation, buffer and poison draws, and batch order. It does **not** change
the splits or the experiences (fixed at seed 42 in Stage 1/1b), so the seed spread measures training
variance only.

## Predictions written down in advance

Idealised weighted-cross-entropy optimum for true-Syn inputs at E1. It assumes the model cannot tell
flipped rows from unflipped ones. Rows: 13,890 LDAP + 19,054 BENIGN + 2,000 buffer BENIGN + 2,000
buffer Syn, of which a share f is stored as BENIGN; K = 3; c_k = sqrt(N / (K n_k)).

p*_Syn = c_S n_S / (c_S n_S + c_B n_flipped)

| f | stored Syn n_S | stored BENIGN n_B | c_S | c_B | p*_Syn |
|---|---|---|---|---|---|
| 0.5 (`main`) | 1,000 | 22,054 | 3.509 | 0.747 | 0.824 (measured in `main`: mean 0.79 on true-Syn buffer rows) |
| 0.75 | 500 | 22,554 | 4.963 | 0.739 | 0.691 |
| 0.9 | 200 | 22,854 | 7.847 | 0.734 | 0.543 |
| 0.95 | 100 | 22,954 | 11.097 | 0.732 | 0.444 |
| 1.0 | 0 | 23,054 | 0 (no Syn label) | 0.731 | 0 |

- **H0 (reproduction).** `repro_s42` makes the same gate decisions as `main`. Exact equality of the
  test recalls is not required on GPU (kernels may be non-deterministic); the max absolute difference
  is reported.
- **H1.** At f = 0.75 and 0.9, the `poisoned` C1 at E1 is accepted in all 3 seeds.
- **H2.** At f = 0.95 and 1.0, the `poisoned` C1 at E1 is rejected (Syn val recall below its floor)
  in at least 2 of 3 seeds.
- **H3 (no direction predicted).** When C1 is rejected, does soft R1 pass the gate?
- **H4.** Forced R1 diagnostic, soft rule: mean weight < 0.05 on flipped rows and > 0.95 on clean
  T-known rows, for every seed.
- **H5 (no direction predicted).** At f = 1.0, does threeway+relabel pass the gate where soft fails?
  Soft can only remove the flipped rows; it cannot bring back any Syn label.
- **H6.** The `clean` arm's C1 is accepted at E1 and E2 in every run (no false rejections).

## Decision and reporting rules

1. Every decision uses val rows only (`s3_continual.py`). Test rows are scored once, after each
   arm's decisions are saved to `rounds.json`.
2. First `python s6_sweep.py --decisions`: val-only table (`sweep/decisions.csv`). It is committed
   to git **before** `python s6_sweep.py --test` is run.
3. Results are reported per seed, plus counts such as "rejected in 2/3 seeds". Means are given with
   min-max. With 3 seeds, no significance test across seeds is claimed.
4. Forced-R1 numbers are always labelled "diagnostic (not the decision rule)".
5. If no C1 is ever rejected, the report says "R1 triggered in 0 of N updates" and makes no claim
   that the repair works.

## Deviations

(none yet)
