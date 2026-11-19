# 13 SEEDS: five more seeds (45-49) for the main defended results

Pre-registration: `experiments/prereg_seeds.md` (commit `a6904bb`, pushed before the run). Status labels as registered:
**the 4-class soft repair at f = 0.75 / 0.9 / 0.95 is a registered replication** (the sweep's pre-registered method);
**the 4-class relabel repair at f = 1.0 (FU2's method) and both 5-class settings (FIVE) are POST HOC** and stay post hoc.

| Step | Where | Commit |
|---|---|---|
| Pre-registration + code | `experiments/prereg_seeds.md`, `s17_seeds.py` | `a6904bb83bc8ea9007c493040ffe7d27e28651fc` |
| Validation (Kaggle `dvdhere/nids-seeds-val` v1, T4, 3,166 s for s17) | `summary.json`, `runs/cii/s4_*`, `runs/cii/s5_*` | decisions commit `309ac649d356d22aabd1034ac0c84f9ecea9a5c9` |
| Held-out test, scored once (Kaggle `dvdhere/nids-seeds-test` v1, CPU) | `test.json` | this commit |

Every file here is a byte-for-byte copy of the Kaggle outputs. `s17 score-test` refused to run unless
`seeds/summary.json` was byte-identical to `summary.json` in `309ac64` and every model's sha256 matched it.
`test.json` records `decisions_commit = 309ac64...`. There was one execution and no crash or resume.

"Learns a class" = held-out test recall > 0 for that class in the deployed model, as in the sweep.

## Registered predictions (seeds 45-49 only)

- **S1: SUPPORTED.** soft repair 15/15; gate only 5/15.
- **S2: SUPPORTED.** relabel repair 5/5; gate only 0/5.
- **S3: SUPPORTED.** soft (f 0.9) 5/5; relabel (f 1.0) 5/5; gate only 0/5 (f 0.9) and 0/5 (f 1.0).
- **S4: SUPPORTED.** max benign FAR of a repaired model 0.0022; clean arms that learned all their classes 30/30.

Gate decisions on validation (`summary.json`): in all 30 poisoned_repair runs, every rejected C1 was followed by an R1
(soft or relabel, as registered), every R1 was accepted (50/50), and every update was deployed (4 classes: 2/2;
5 classes: 3/3).

## Pooled with seeds 42-44 (descriptive)

Seeds 42-44 come from the committed files `04_sweep/deployed_models.csv` (4 classes; gate only, soft repair, clean),
`06_fu2/test/test_recall.csv` (4 classes, relabel repair at f = 1.0) and `12_five/test.json` (5 classes). The
test rows are identical across all runs (4 classes: 46,527 rows, rows_sha256 `09245b03b63d...`; 5 classes: 58,516
rows, `b7cb85fb7490...`; checked). Recall, macro recall and benign FAR are means over all 8 seeds. Accuracy for seeds
42-44 is derived exactly as sum over classes of recall x support / rows (same test rows). Supports come from the
confusion matrices in `test.json`: BENIGN 19,028; Syn 11,946; LDAP 4,576; UDP 10,977; TFTP 11,989.

| Setting | Arm | All classes learned, 45-49 | Pooled 42-49 | Accuracy | Macro recall | Benign FAR mean (max) |
|---|---|---|---|---|---|---|
| 4 classes, f = 0.75 | clean | 5/5 | 8/8 | 0.9962 | 0.9939 | 0.0009 (0.0018) |
| 4 classes, f = 0.75 | gate only | 5/5 | 7/8 | 0.9542 | 0.9313 | 0.0007 (0.0013) |
| 4 classes, f = 0.75 | soft repair | 5/5 | 8/8 | 0.9953 | 0.9927 | 0.0009 (0.0013) |
| 4 classes, f = 0.9 | clean | 5/5 | 8/8 | 0.9962 | 0.9939 | 0.0009 (0.0018) |
| 4 classes, f = 0.9 | gate only | 0/5 | 1/8 | 0.8615 | 0.7440 | 0.0009 (0.0048) |
| 4 classes, f = 0.9 | soft repair | 5/5 | 8/8 | 0.9957 | 0.9933 | 0.0008 (0.0013) |
| 4 classes, f = 0.95 | clean | 5/5 | 8/8 | 0.9962 | 0.9939 | 0.0009 (0.0018) |
| 4 classes, f = 0.95 | gate only | 0/5 | 0/8 | 0.6762 | 0.5293 | 0.0005 (0.0038) |
| 4 classes, f = 0.95 | soft repair | 5/5 | 8/8 | 0.9945 | 0.9919 | 0.0011 (0.0018) |
| 4 classes, f = 1.0 | clean | 5/5 | 8/8 | 0.9962 | 0.9939 | 0.0009 (0.0018) |
| 4 classes, f = 1.0 | gate only | 0/5 | 0/8 | 0.6645 | 0.4988 | 0.0000 (0.0001) |
| 4 classes, f = 1.0 | relabel repair | 5/5 | 8/8 | 0.9961 | 0.9938 | 0.0009 (0.0016) |
| 5 classes, f = 0.9 | clean | 5/5 | 8/8 | 0.9961 | 0.9943 | 0.0012 (0.0015) |
| 5 classes, f = 0.9 | gate only | 0/5 | 0/8 | 0.6226 | 0.4974 | 0.0002 (0.0009) |
| 5 classes, f = 0.9 | soft repair | 5/5 | 8/8 | 0.9937 | 0.9920 | 0.0015 (0.0022) |
| 5 classes, f = 1.0 | clean | 5/5 | 8/8 | 0.9961 | 0.9943 | 0.0012 (0.0015) |
| 5 classes, f = 1.0 | gate only | 0/5 | 0/8 | 0.5284 | 0.3991 | 0.0000 (0.0001) |
| 5 classes, f = 1.0 | relabel repair | 5/5 | 8/8 | 0.9960 | 0.9939 | 0.0012 (0.0020) |

Per-class recall (R, mean over seeds 42-49) and precision (P, mean over seeds 45-49 where the class is predicted).
Precision was not committed for seeds 42-44 of the sweep and FIVE. n/a means the class was never predicted.

| Setting | Arm | BENIGN R / P | Syn R / P | LDAP R / P | UDP R / P | TFTP R / P |
|---|---|---|---|---|---|---|
| 4 classes, f = 0.75 | clean | 0.9991 / 0.9941 | 0.9949 / 0.9991 | 0.9839 / 0.9906 | 0.9979 / 0.9982 |  |
| 4 classes, f = 0.75 | gate only | 0.9993 / 0.9919 | 0.9933 / 0.9991 | 0.8595 / 0.9926 | 0.8732 / 0.9979 |  |
| 4 classes, f = 0.75 | soft repair | 0.9991 / 0.9919 | 0.9914 / 0.9991 | 0.9822 / 0.9926 | 0.9981 / 0.9979 |  |
| 4 classes, f = 0.9 | clean | 0.9991 / 0.9941 | 0.9949 / 0.9991 | 0.9839 / 0.9906 | 0.9979 / 0.9982 |  |
| 4 classes, f = 0.9 | gate only | 0.9991 / 0.9117 | 0.9806 / 0.9994 | 0.2467 / 0.8905 | 0.7495 / 0.7113 |  |
| 4 classes, f = 0.9 | soft repair | 0.9992 / 0.9936 | 0.9926 / 0.9993 | 0.9830 / 0.9902 | 0.9983 / 0.9981 |  |
| 4 classes, f = 0.95 | clean | 0.9991 / 0.9941 | 0.9949 / 0.9991 | 0.9839 / 0.9906 | 0.9979 / 0.9982 |  |
| 4 classes, f = 0.95 | gate only | 0.9995 / 0.5661 | 0.9941 / 0.9985 | 0.1237 / 0.9529 | 0.0000 / n/a |  |
| 4 classes, f = 0.95 | soft repair | 0.9989 / 0.9927 | 0.9905 / 0.9991 | 0.9810 / 0.9797 | 0.9970 / 0.9973 |  |
| 4 classes, f = 1.0 | clean | 0.9991 / 0.9941 | 0.9949 / 0.9991 | 0.9839 / 0.9906 | 0.9979 / 0.9982 |  |
| 4 classes, f = 1.0 | gate only | 1.0000 / 0.5496 | 0.9953 / 0.9984 | 0.0000 / n/a | 0.0000 / n/a |  |
| 4 classes, f = 1.0 | relabel repair | 0.9991 / 0.9941 | 0.9944 / 0.9993 | 0.9836 / 0.9907 | 0.9982 / 0.9983 |  |
| 5 classes, f = 0.9 | clean | 0.9988 / 0.9936 | 0.9950 / 0.9979 | 0.9840 / 0.9907 | 0.9975 / 0.9985 | 0.9962 / 0.9984 |
| 5 classes, f = 0.9 | gate only | 0.9998 / 0.4579 | 0.9881 / 0.9966 | 0.0000 / n/a | 0.3744 / 0.6238 | 0.1246 / n/a |
| 5 classes, f = 0.9 | soft repair | 0.9985 / 0.9902 | 0.9846 / 0.9983 | 0.9837 / 0.9860 | 0.9967 / 0.9985 | 0.9962 / 0.9971 |
| 5 classes, f = 1.0 | clean | 0.9988 / 0.9936 | 0.9950 / 0.9979 | 0.9840 / 0.9907 | 0.9975 / 0.9985 | 0.9962 / 0.9984 |
| 5 classes, f = 1.0 | gate only | 1.0000 / 0.4085 | 0.9953 / 0.9965 | 0.0000 / n/a | 0.0000 / n/a | 0.0000 / n/a |
| 5 classes, f = 1.0 | relabel repair | 0.9988 / 0.9930 | 0.9950 / 0.9977 | 0.9816 / 0.9920 | 0.9982 / 0.9981 | 0.9962 / 0.9987 |

## Every deployed model of seeds 45-49 (held-out test)

<details><summary>90 deployed models: accuracy, macro recall, benign FAR, per-class recall and precision</summary>

| Run | Arm | Classes | Accuracy | Macro recall | Benign FAR | Recall (per class) | Precision (per class) |
|---|---|---|---|---|---|---|---|
| s4_f0.75_s45 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9939 | 0.0002 | BENIGN 0.9998, Syn 0.9951, LDAP 0.9832, UDP 0.9974 | BENIGN 0.9937, Syn 0.9990, LDAP 0.9938, UDP 0.9994 |
| s4_f0.75_s45 | poisoned | BENIGN, Syn, LDAP, UDP | 0.9945 | 0.9927 | 0.0010 | BENIGN 0.9990, Syn 0.9885, LDAP 0.9869, UDP 0.9964 | BENIGN 0.9902, Syn 0.9993, LDAP 0.9878, UDP 0.9997 |
| s4_f0.75_s45 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9945 | 0.9927 | 0.0010 | BENIGN 0.9990, Syn 0.9885, LDAP 0.9869, UDP 0.9964 | BENIGN 0.9902, Syn 0.9993, LDAP 0.9878, UDP 0.9997 |
| s4_f0.9_s45 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9939 | 0.0002 | BENIGN 0.9998, Syn 0.9951, LDAP 0.9832, UDP 0.9974 | BENIGN 0.9937, Syn 0.9990, LDAP 0.9938, UDP 0.9994 |
| s4_f0.9_s45 | poisoned | BENIGN, Syn, UDP | 0.8993 | 0.7478 | 0.0004 | BENIGN 0.9996, Syn 0.9920, LDAP 0.0000, UDP 0.9995 | BENIGN 0.9888, Syn 0.9990, UDP 0.7111 |
| s4_f0.9_s45 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9956 | 0.9923 | 0.0007 | BENIGN 0.9993, Syn 0.9933, LDAP 0.9773, UDP 0.9995 | BENIGN 0.9928, Syn 0.9989, LDAP 0.9978, UDP 0.9961 |
| s4_f0.95_s45 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9939 | 0.0002 | BENIGN 0.9998, Syn 0.9951, LDAP 0.9832, UDP 0.9974 | BENIGN 0.9937, Syn 0.9990, LDAP 0.9938, UDP 0.9994 |
| s4_f0.95_s45 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5495, Syn 0.9989 |
| s4_f0.95_s45 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9931 | 0.9908 | 0.0013 | BENIGN 0.9987, Syn 0.9898, LDAP 0.9838, UDP 0.9911 | BENIGN 0.9911, Syn 0.9988, LDAP 0.9772, UDP 0.9973 |
| s4_f1.0_s45 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9939 | 0.0002 | BENIGN 0.9998, Syn 0.9951, LDAP 0.9832, UDP 0.9974 | BENIGN 0.9937, Syn 0.9990, LDAP 0.9938, UDP 0.9994 |
| s4_f1.0_s45 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5495, Syn 0.9989 |
| s4_f1.0_s45 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9967 | 0.9938 | 0.0001 | BENIGN 0.9999, Syn 0.9951, LDAP 0.9808, UDP 0.9994 | BENIGN 0.9932, Syn 0.9990, LDAP 0.9993, UDP 0.9991 |
| s5_f0.9_s45 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9965 | 0.9949 | 0.0015 | BENIGN 0.9985, Syn 0.9951, LDAP 0.9854, UDP 0.9992, TFTP 0.9964 | BENIGN 0.9942, Syn 0.9978, LDAP 0.9936, UDP 0.9985, TFTP 0.9979 |
| s5_f0.9_s45 | poisoned | BENIGN, Syn, UDP | 0.7157 | 0.5989 | 0.0002 | BENIGN 0.9998, Syn 0.9951, LDAP 0.0000, UDP 0.9994, TFTP 0.0000 | BENIGN 0.6558, Syn 0.9974, UDP 0.6238 |
| s5_f0.9_s45 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9956 | 0.9941 | 0.0022 | BENIGN 0.9978, Syn 0.9928, LDAP 0.9854, UDP 0.9984, TFTP 0.9961 | BENIGN 0.9926, Syn 0.9976, LDAP 0.9912, UDP 0.9989, TFTP 0.9969 |
| s5_f1.0_s45 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9965 | 0.9949 | 0.0015 | BENIGN 0.9985, Syn 0.9951, LDAP 0.9854, UDP 0.9992, TFTP 0.9964 | BENIGN 0.9942, Syn 0.9978, LDAP 0.9936, UDP 0.9985, TFTP 0.9979 |
| s5_f1.0_s45 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0000 | BENIGN 1.0000, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4084, Syn 0.9969 |
| s5_f1.0_s45 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9964 | 0.9943 | 0.0009 | BENIGN 0.9991, Syn 0.9951, LDAP 0.9816, UDP 0.9992, TFTP 0.9964 | BENIGN 0.9938, Syn 0.9978, LDAP 0.9960, UDP 0.9979, TFTP 0.9978 |
| s4_f0.75_s46 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9941 | 0.0005 | BENIGN 0.9995, Syn 0.9953, LDAP 0.9847, UDP 0.9970 | BENIGN 0.9943, Syn 0.9988, LDAP 0.9921, UDP 0.9991 |
| s4_f0.75_s46 | poisoned | BENIGN, Syn, LDAP, UDP | 0.9956 | 0.9928 | 0.0008 | BENIGN 0.9992, Syn 0.9933, LDAP 0.9810, UDP 0.9978 | BENIGN 0.9921, Syn 0.9989, LDAP 0.9934, UDP 0.9988 |
| s4_f0.75_s46 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9956 | 0.9928 | 0.0008 | BENIGN 0.9992, Syn 0.9933, LDAP 0.9810, UDP 0.9978 | BENIGN 0.9921, Syn 0.9989, LDAP 0.9934, UDP 0.9988 |
| s4_f0.9_s46 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9941 | 0.0005 | BENIGN 0.9995, Syn 0.9953, LDAP 0.9847, UDP 0.9970 | BENIGN 0.9943, Syn 0.9988, LDAP 0.9921, UDP 0.9991 |
| s4_f0.9_s46 | poisoned | BENIGN, Syn, UDP | 0.8916 | 0.7402 | 0.0001 | BENIGN 0.9999, Syn 0.9624, LDAP 0.0000, UDP 0.9985 | BENIGN 0.9691, Syn 0.9997, UDP 0.7120 |
| s4_f0.9_s46 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9952 | 0.9937 | 0.0004 | BENIGN 0.9996, Syn 0.9890, LDAP 0.9893, UDP 0.9968 | BENIGN 0.9945, Syn 1.0000, LDAP 0.9767, UDP 0.9991 |
| s4_f0.95_s46 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9941 | 0.0005 | BENIGN 0.9995, Syn 0.9953, LDAP 0.9847, UDP 0.9970 | BENIGN 0.9943, Syn 0.9988, LDAP 0.9921, UDP 0.9991 |
| s4_f0.95_s46 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0001 | BENIGN 0.9999, Syn 0.9954, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5497, Syn 0.9977 |
| s4_f0.95_s46 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9960 | 0.9921 | 0.0002 | BENIGN 0.9998, Syn 0.9951, LDAP 0.9742, UDP 0.9992 | BENIGN 0.9923, Syn 0.9981, LDAP 0.9991, UDP 0.9987 |
| s4_f1.0_s46 | clean | BENIGN, Syn, LDAP, UDP | 0.9964 | 0.9941 | 0.0005 | BENIGN 0.9995, Syn 0.9953, LDAP 0.9847, UDP 0.9970 | BENIGN 0.9943, Syn 0.9988, LDAP 0.9921, UDP 0.9991 |
| s4_f1.0_s46 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0001 | BENIGN 0.9999, Syn 0.9954, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5497, Syn 0.9977 |
| s4_f1.0_s46 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9965 | 0.9944 | 0.0006 | BENIGN 0.9994, Syn 0.9937, LDAP 0.9854, UDP 0.9992 | BENIGN 0.9944, Syn 0.9992, LDAP 0.9925, UDP 0.9988 |
| s5_f0.9_s46 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9964 | 0.9952 | 0.0011 | BENIGN 0.9989, Syn 0.9935, LDAP 0.9884, UDP 0.9989, TFTP 0.9961 | BENIGN 0.9937, Syn 0.9986, LDAP 0.9880, UDP 0.9986, TFTP 0.9997 |
| s5_f0.9_s46 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0001 | BENIGN 0.9999, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4085, Syn 0.9961 |
| s5_f0.9_s46 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9923 | 0.9904 | 0.0007 | BENIGN 0.9993, Syn 0.9754, LDAP 0.9825, UDP 0.9986, TFTP 0.9962 | BENIGN 0.9810, Syn 0.9989, LDAP 0.9927, UDP 0.9980, TFTP 0.9990 |
| s5_f1.0_s46 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9964 | 0.9952 | 0.0011 | BENIGN 0.9989, Syn 0.9935, LDAP 0.9884, UDP 0.9989, TFTP 0.9961 | BENIGN 0.9937, Syn 0.9986, LDAP 0.9880, UDP 0.9986, TFTP 0.9997 |
| s5_f1.0_s46 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0001 | BENIGN 0.9999, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4085, Syn 0.9961 |
| s5_f1.0_s46 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9964 | 0.9949 | 0.0016 | BENIGN 0.9984, Syn 0.9950, LDAP 0.9860, UDP 0.9991, TFTP 0.9960 | BENIGN 0.9934, Syn 0.9981, LDAP 0.9923, UDP 0.9979, TFTP 0.9995 |
| s4_f0.75_s47 | clean | BENIGN, Syn, LDAP, UDP | 0.9960 | 0.9931 | 0.0004 | BENIGN 0.9996, Syn 0.9933, LDAP 0.9803, UDP 0.9992 | BENIGN 0.9930, Syn 0.9996, LDAP 0.9925, UDP 0.9987 |
| s4_f0.75_s47 | poisoned | BENIGN, Syn, LDAP, UDP | 0.9955 | 0.9914 | 0.0004 | BENIGN 0.9996, Syn 0.9946, LDAP 0.9722, UDP 0.9993 | BENIGN 0.9925, Syn 0.9989, LDAP 0.9987, UDP 0.9959 |
| s4_f0.75_s47 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9955 | 0.9914 | 0.0004 | BENIGN 0.9996, Syn 0.9946, LDAP 0.9722, UDP 0.9993 | BENIGN 0.9925, Syn 0.9989, LDAP 0.9987, UDP 0.9959 |
| s4_f0.9_s47 | clean | BENIGN, Syn, LDAP, UDP | 0.9960 | 0.9931 | 0.0004 | BENIGN 0.9996, Syn 0.9933, LDAP 0.9803, UDP 0.9992 | BENIGN 0.9930, Syn 0.9996, LDAP 0.9925, UDP 0.9987 |
| s4_f0.9_s47 | poisoned | BENIGN, Syn, UDP | 0.9000 | 0.7485 | 0.0005 | BENIGN 0.9995, Syn 0.9951, LDAP 0.0000, UDP 0.9994 | BENIGN 0.9906, Syn 0.9989, UDP 0.7110 |
| s4_f0.9_s47 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9960 | 0.9929 | 0.0010 | BENIGN 0.9990, Syn 0.9951, LDAP 0.9788, UDP 0.9988 | BENIGN 0.9930, Syn 0.9989, LDAP 0.9949, UDP 0.9985 |
| s4_f0.95_s47 | clean | BENIGN, Syn, LDAP, UDP | 0.9960 | 0.9931 | 0.0004 | BENIGN 0.9996, Syn 0.9933, LDAP 0.9803, UDP 0.9992 | BENIGN 0.9930, Syn 0.9996, LDAP 0.9925, UDP 0.9987 |
| s4_f0.95_s47 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9954, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5498, Syn 0.9977 |
| s4_f0.95_s47 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9900 | 0.9875 | 0.0012 | BENIGN 0.9988, Syn 0.9735, LDAP 0.9814, UDP 0.9964 | BENIGN 0.9932, Syn 0.9998, LDAP 0.9372, UDP 0.9973 |
| s4_f1.0_s47 | clean | BENIGN, Syn, LDAP, UDP | 0.9960 | 0.9931 | 0.0004 | BENIGN 0.9996, Syn 0.9933, LDAP 0.9803, UDP 0.9992 | BENIGN 0.9930, Syn 0.9996, LDAP 0.9925, UDP 0.9987 |
| s4_f1.0_s47 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9954, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5498, Syn 0.9977 |
| s4_f1.0_s47 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9952 | 0.9934 | 0.0016 | BENIGN 0.9984, Syn 0.9923, LDAP 0.9865, UDP 0.9964 | BENIGN 0.9940, Syn 0.9997, LDAP 0.9815, UDP 0.9982 |
| s5_f0.9_s47 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9963 | 0.9941 | 0.0008 | BENIGN 0.9992, Syn 0.9953, LDAP 0.9803, UDP 0.9993, TFTP 0.9963 | BENIGN 0.9934, Syn 0.9976, LDAP 0.9969, UDP 0.9973, TFTP 0.9988 |
| s5_f0.9_s47 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0000 | BENIGN 1.0000, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4085, Syn 0.9962 |
| s5_f0.9_s47 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9955 | 0.9938 | 0.0014 | BENIGN 0.9986, Syn 0.9908, LDAP 0.9838, UDP 0.9991, TFTP 0.9965 | BENIGN 0.9933, Syn 0.9980, LDAP 0.9832, UDP 0.9985, TFTP 0.9986 |
| s5_f1.0_s47 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9963 | 0.9941 | 0.0008 | BENIGN 0.9992, Syn 0.9953, LDAP 0.9803, UDP 0.9993, TFTP 0.9963 | BENIGN 0.9934, Syn 0.9976, LDAP 0.9969, UDP 0.9973, TFTP 0.9988 |
| s5_f1.0_s47 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0000 | BENIGN 1.0000, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4085, Syn 0.9962 |
| s5_f1.0_s47 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9957 | 0.9926 | 0.0010 | BENIGN 0.9990, Syn 0.9954, LDAP 0.9736, UDP 0.9988, TFTP 0.9964 | BENIGN 0.9927, Syn 0.9974, LDAP 0.9962, UDP 0.9958, TFTP 0.9985 |
| s4_f0.75_s48 | clean | BENIGN, Syn, LDAP, UDP | 0.9961 | 0.9937 | 0.0007 | BENIGN 0.9993, Syn 0.9951, LDAP 0.9834, UDP 0.9970 | BENIGN 0.9944, Syn 0.9988, LDAP 0.9914, UDP 0.9982 |
| s4_f0.75_s48 | poisoned | BENIGN, Syn, LDAP, UDP | 0.9954 | 0.9929 | 0.0009 | BENIGN 0.9991, Syn 0.9931, LDAP 0.9821, UDP 0.9972 | BENIGN 0.9923, Syn 0.9989, LDAP 0.9921, UDP 0.9984 |
| s4_f0.75_s48 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9954 | 0.9929 | 0.0009 | BENIGN 0.9991, Syn 0.9931, LDAP 0.9821, UDP 0.9972 | BENIGN 0.9923, Syn 0.9989, LDAP 0.9921, UDP 0.9984 |
| s4_f0.9_s48 | clean | BENIGN, Syn, LDAP, UDP | 0.9961 | 0.9937 | 0.0007 | BENIGN 0.9993, Syn 0.9951, LDAP 0.9834, UDP 0.9970 | BENIGN 0.9944, Syn 0.9988, LDAP 0.9914, UDP 0.9982 |
| s4_f0.9_s48 | poisoned | BENIGN, Syn, UDP | 0.8917 | 0.7404 | 0.0006 | BENIGN 0.9994, Syn 0.9628, LDAP 0.0000, UDP 0.9994 | BENIGN 0.9705, Syn 0.9996, UDP 0.7112 |
| s4_f0.9_s48 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9962 | 0.9931 | 0.0003 | BENIGN 0.9997, Syn 0.9951, LDAP 0.9792, UDP 0.9984 | BENIGN 0.9933, Syn 0.9990, LDAP 0.9962, UDP 0.9984 |
| s4_f0.95_s48 | clean | BENIGN, Syn, LDAP, UDP | 0.9961 | 0.9937 | 0.0007 | BENIGN 0.9993, Syn 0.9951, LDAP 0.9834, UDP 0.9970 | BENIGN 0.9944, Syn 0.9988, LDAP 0.9914, UDP 0.9982 |
| s4_f0.95_s48 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5495, Syn 0.9989 |
| s4_f0.95_s48 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9956 | 0.9924 | 0.0006 | BENIGN 0.9994, Syn 0.9951, LDAP 0.9781, UDP 0.9968 | BENIGN 0.9935, Syn 0.9989, LDAP 0.9918, UDP 0.9973 |
| s4_f1.0_s48 | clean | BENIGN, Syn, LDAP, UDP | 0.9961 | 0.9937 | 0.0007 | BENIGN 0.9993, Syn 0.9951, LDAP 0.9834, UDP 0.9970 | BENIGN 0.9944, Syn 0.9988, LDAP 0.9914, UDP 0.9982 |
| s4_f1.0_s48 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5495, Syn 0.9989 |
| s4_f1.0_s48 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9967 | 0.9944 | 0.0008 | BENIGN 0.9992, Syn 0.9951, LDAP 0.9840, UDP 0.9994 | BENIGN 0.9945, Syn 0.9989, LDAP 0.9973, UDP 0.9979 |
| s5_f0.9_s48 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9958 | 0.9938 | 0.0014 | BENIGN 0.9986, Syn 0.9951, LDAP 0.9823, UDP 0.9972, TFTP 0.9961 | BENIGN 0.9934, Syn 0.9978, LDAP 0.9916, UDP 0.9984, TFTP 0.9971 |
| s5_f0.9_s48 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0001 | BENIGN 0.9999, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4085, Syn 0.9959 |
| s5_f0.9_s48 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9940 | 0.9926 | 0.0020 | BENIGN 0.9980, Syn 0.9898, LDAP 0.9862, UDP 0.9931, TFTP 0.9958 | BENIGN 0.9901, Syn 0.9986, LDAP 0.9777, UDP 0.9984, TFTP 0.9981 |
| s5_f1.0_s48 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9958 | 0.9938 | 0.0014 | BENIGN 0.9986, Syn 0.9951, LDAP 0.9823, UDP 0.9972, TFTP 0.9961 | BENIGN 0.9934, Syn 0.9978, LDAP 0.9916, UDP 0.9984, TFTP 0.9971 |
| s5_f1.0_s48 | poisoned | BENIGN, Syn | 0.5284 | 0.3991 | 0.0001 | BENIGN 0.9999, Syn 0.9954, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4085, Syn 0.9959 |
| s5_f1.0_s48 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9960 | 0.9938 | 0.0005 | BENIGN 0.9995, Syn 0.9951, LDAP 0.9812, UDP 0.9971, TFTP 0.9961 | BENIGN 0.9926, Syn 0.9976, LDAP 0.9912, UDP 0.9992, TFTP 0.9990 |
| s4_f0.75_s49 | clean | BENIGN, Syn, LDAP, UDP | 0.9953 | 0.9931 | 0.0018 | BENIGN 0.9982, Syn 0.9941, LDAP 0.9840, UDP 0.9962 | BENIGN 0.9953, Syn 0.9994, LDAP 0.9832, UDP 0.9957 |
| s4_f0.75_s49 | poisoned | BENIGN, Syn, LDAP, UDP | 0.9950 | 0.9926 | 0.0008 | BENIGN 0.9992, Syn 0.9914, LDAP 0.9832, UDP 0.9968 | BENIGN 0.9922, Syn 0.9995, LDAP 0.9912, UDP 0.9968 |
| s4_f0.75_s49 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9950 | 0.9926 | 0.0008 | BENIGN 0.9992, Syn 0.9914, LDAP 0.9832, UDP 0.9968 | BENIGN 0.9922, Syn 0.9995, LDAP 0.9912, UDP 0.9968 |
| s4_f0.9_s49 | clean | BENIGN, Syn, LDAP, UDP | 0.9953 | 0.9931 | 0.0018 | BENIGN 0.9982, Syn 0.9941, LDAP 0.9840, UDP 0.9962 | BENIGN 0.9953, Syn 0.9994, LDAP 0.9832, UDP 0.9957 |
| s4_f0.9_s49 | poisoned | BENIGN, Syn, LDAP | 0.7585 | 0.7438 | 0.0048 | BENIGN 0.9952, Syn 0.9901, LDAP 0.9897, UDP 0.0000 | BENIGN 0.6395, Syn 0.9997, LDAP 0.8905 |
| s4_f0.9_s49 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9958 | 0.9938 | 0.0006 | BENIGN 0.9994, Syn 0.9931, LDAP 0.9865, UDP 0.9964 | BENIGN 0.9944, Syn 0.9997, LDAP 0.9854, UDP 0.9984 |
| s4_f0.95_s49 | clean | BENIGN, Syn, LDAP, UDP | 0.9953 | 0.9931 | 0.0018 | BENIGN 0.9982, Syn 0.9941, LDAP 0.9840, UDP 0.9962 | BENIGN 0.9953, Syn 0.9994, LDAP 0.9832, UDP 0.9957 |
| s4_f0.95_s49 | poisoned | BENIGN, Syn, LDAP | 0.7579 | 0.7430 | 0.0038 | BENIGN 0.9962, Syn 0.9861, LDAP 0.9897, UDP 0.0000 | BENIGN 0.6321, Syn 0.9993, LDAP 0.9529 |
| s4_f0.95_s49 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9956 | 0.9933 | 0.0010 | BENIGN 0.9990, Syn 0.9914, LDAP 0.9836, UDP 0.9994 | BENIGN 0.9936, Syn 0.9997, LDAP 0.9934, UDP 0.9957 |
| s4_f1.0_s49 | clean | BENIGN, Syn, LDAP, UDP | 0.9953 | 0.9931 | 0.0018 | BENIGN 0.9982, Syn 0.9941, LDAP 0.9840, UDP 0.9962 | BENIGN 0.9953, Syn 0.9994, LDAP 0.9832, UDP 0.9957 |
| s4_f1.0_s49 | poisoned | BENIGN, Syn | 0.6645 | 0.4988 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000 | BENIGN 0.5495, Syn 0.9989 |
| s4_f1.0_s49 | poisoned_repair | BENIGN, Syn, LDAP, UDP | 0.9954 | 0.9932 | 0.0012 | BENIGN 0.9988, Syn 0.9934, LDAP 0.9845, UDP 0.9961 | BENIGN 0.9945, Syn 0.9997, LDAP 0.9828, UDP 0.9976 |
| s5_f0.9_s49 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9957 | 0.9939 | 0.0014 | BENIGN 0.9986, Syn 0.9951, LDAP 0.9840, UDP 0.9951, TFTP 0.9965 | BENIGN 0.9932, Syn 0.9978, LDAP 0.9832, UDP 0.9995, TFTP 0.9987 |
| s5_f0.9_s49 | poisoned | BENIGN, Syn | 0.5283 | 0.3990 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4084, Syn 0.9973 |
| s5_f0.9_s49 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9948 | 0.9932 | 0.0018 | BENIGN 0.9982, Syn 0.9932, LDAP 0.9858, UDP 0.9925, TFTP 0.9965 | BENIGN 0.9940, Syn 0.9985, LDAP 0.9851, UDP 0.9985, TFTP 0.9927 |
| s5_f1.0_s49 | clean | BENIGN, Syn, LDAP, UDP, TFTP | 0.9957 | 0.9939 | 0.0014 | BENIGN 0.9986, Syn 0.9951, LDAP 0.9840, UDP 0.9951, TFTP 0.9965 | BENIGN 0.9932, Syn 0.9978, LDAP 0.9832, UDP 0.9995, TFTP 0.9987 |
| s5_f1.0_s49 | poisoned | BENIGN, Syn | 0.5283 | 0.3990 | 0.0000 | BENIGN 1.0000, Syn 0.9951, LDAP 0.0000, UDP 0.0000, TFTP 0.0000 | BENIGN 0.4084, Syn 0.9973 |
| s5_f1.0_s49 | poisoned_repair | BENIGN, Syn, LDAP, UDP, TFTP | 0.9955 | 0.9935 | 0.0013 | BENIGN 0.9987, Syn 0.9951, LDAP 0.9823, UDP 0.9950, TFTP 0.9965 | BENIGN 0.9926, Syn 0.9978, LDAP 0.9845, UDP 0.9996, TFTP 0.9985 |

</details>

## Scope

The 4-class soft setting is a registered replication; relabel repair and the 5-class settings are POST HOC. One
attacked class (stored Syn -> BENIGN, once before E1), seeds 45-49 fixed in advance, no significance test (as
registered). The pooled table is descriptive.
