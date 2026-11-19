# 12 FIVE: the four-class defence with a fifth class, TFTP (POST HOC)

**This whole study is POST HOC.** TFTP was chosen after the earlier results showed it is easy to learn when clean.
Pre-registration: `experiments/prereg_five.md` (commit `fccbf6a`, pushed before any FIVE run). Nothing in the
original claims changes.

| Step | Where | Commit |
|---|---|---|
| Pre-registration + code | `experiments/prereg_five.md`, `s16_five.py` | `fccbf6ae45c1db7e138851ceaf9330c986f38491` |
| Clean pilot + validation (Kaggle `dvdhere/nids-five-val` v1, T4, 780 s) | `pilot.json`, `summary.json`, `schedule.json`, `cii/experience_manifest.json`, `runs/cii/five_*` | decisions commit `e6a2af42439054ec620c200688b4cbd71052fe15` |
| Held-out test, scored once (Kaggle `dvdhere/nids-five-test` v1, CPU) | `test.json` | this commit |

Every file here is a byte-for-byte copy of the Kaggle outputs. `s16 score-test` refused to run unless
`five/summary.json` was byte-identical to `summary.json` in `e6a2af4` and every model's sha256 matched it.
`test.json` records `decisions_commit = e6a2af4...`.

Schedule (tag `five`, CII): E0 = BENIGN + Syn, E1 + LDAP, E2 + UDP, E3 + TFTP. Training rows per experience are
13,890 attack rows and 14,291 benign rows (14,290 at E3), from `cii/experience_manifest.json`. Seeds 42, 43, 44.
f = 0.9 uses soft R1; f = 1.0 uses threeway+relabel R1.

## Clean pilot (stopping rule)

`pilot.json`: **proceed = true.** All 3 clean updates were accepted in each of seeds 42, 43 and 44 (9/9, no gate
reasons), so the full run went ahead.

## Results

Updates and R1 counts come from `summary.json`. Test metrics come from `test.json` (one test set: 19,028 benign
rows). "All 5" means the deployed model's classes include all five and each has test recall > 0. Recall and FAR are
means over the 3 seeds.

| f | Arm | Updates accepted (per seed) | R1 passed/ran | All 5 classes | BENIGN | Syn | LDAP | UDP | TFTP | Benign FAR (per seed) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | clean | 3, 3, 3 | 0/0 | 3/3 | 0.9990 | 0.9953 | 0.9838 | 0.9969 | 0.9961 | 0.0010 (0.0009, 0.0007, 0.0014) |
| 0.9 | poisoned (gate only) | 0, 1, 2 | 0/0 | 0/3 | 0.9996 | 0.9760 | 0.0000 | 0.6654 | 0.3322 | 0.0004 (0.0000, 0.0009, 0.0004) |
| 0.9 | poisoned_repair (soft R1) | 3, 3, 3 | 5/5 | 3/3 | 0.9988 | 0.9782 | 0.9820 | 0.9972 | 0.9963 | 0.0012 (0.0007, 0.0015, 0.0013) |
| 1.0 | clean | 3, 3, 3 | 0/0 | 3/3 | 0.9990 | 0.9953 | 0.9838 | 0.9969 | 0.9961 | 0.0010 (0.0009, 0.0007, 0.0014) |
| 1.0 | poisoned (gate only) | 0, 0, 0 | 0/0 | 0/3 | 1.0000 | 0.9953 | 0.0000 | 0.0000 | 0.0000 | 0.0000 (0.0000, 0.0000, 0.0000) |
| 1.0 | poisoned_repair (relabel R1) | 3, 3, 3 | 9/9 | 3/3 | 0.9985 | 0.9947 | 0.9828 | 0.9987 | 0.9959 | 0.0015 (0.0010, 0.0016, 0.0020) |

The clean arm is the same at both f, because the attack fraction does not affect it.

At f = 0.9 the gate-only arm rejected E1 (LDAP) in every seed but accepted later updates in s43 (E2) and s44 (E2, E3).
Its deployed classes are therefore [BENIGN, Syn], [BENIGN, Syn, UDP] and [BENIGN, Syn, UDP, TFTP]. Soft repair's
lowest test Syn recall was 0.9622 (s42).

## Hypotheses (POST HOC)

- **F1: SUPPORTED.** Every clean update was accepted (3 updates x 3 seeds = 9/9), and the clean deployed model
  learned all 5 classes in 3/3 seeds (mean test recall: LDAP 0.9838, UDP 0.9969, TFTP 0.9961).
- **F2: SUPPORTED.** At f = 0.9, soft repair deployed a model with all 5 classes in 3/3 seeds (R1 5/5 accepted).
  Gate only did so in 0/3 seeds (fewer than 3/3); LDAP test recall was 0.0 in every seed.
- **F3: SUPPORTED.** At f = 1.0, relabel repair learned all 5 classes in 3/3 seeds (R1 9/9 accepted). Gate only did
  so in 0/3: it accepted no update, so LDAP, UDP and TFTP test recall were 0.0 in every seed.
- **F4: SUPPORTED.** The benign FAR of every repaired deployed model was <= 0.005 on held-out test. Soft (f 0.9):
  0.0007, 0.0015, 0.0013. Relabel (f 1.0): 0.0010, 0.0016, 0.0020. The maximum was 0.0020 (19,028 benign test rows).

## Scope

POST HOC, one attacked class (stored Syn → BENIGN, once before E1), one added class (TFTP), 3 seeds, no significance
claim. One execution; no other class, schedule or threshold was tried. Neither kernel crashed and nothing was resumed.
