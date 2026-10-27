# FU2 results (notebook 06): deployed teacher relabelling at f = 1.0, seeds 42-44

Pre-registration: `experiments/prereg_followup_fu2.md` (commit `0d9a807`). The validation files below were committed
unchanged in `db4ca21` **before** the test was scored. The test was scored once afterwards; see "Held-out test" at the end.

## Provenance (read before any number below)

Deploying relabelling, and the fraction f = 1.0, were chosen **after** seeing the sweep's forced diagnostic
(threeway+relabel "would pass" 6/6, `results/04_sweep/diagnostics.csv`). The 0.9 / 0.1 thresholds were chosen after
seeing `main`. **E1 replays that diagnostic** (same start model, buffer, seed and code path; P0 checks below), so E1 is
**not independent evidence**. The new evidence is E2, which runs after relabelling was deployed at E1, and the
held-out test, which is still to come. CII only, one attacked class (Syn -> BENIGN), and the teacher knows only
BENIGN and Syn.

## Files

`fu2/cii/fu2_relabel_f1.0_s{42,43,44}/summary.json` and `.../poisoned_relabel/rounds.json` are byte-identical to
the contents of `fu2_val.zip` (sha256 `7153065a78fadacab21230b10a2e9877be140ce6236f09a0fceccdb12d7048eb`), committed
unchanged. Run on Kaggle (T4, `cuda`, torch 2.11.0+cu128) at code commit `6e2de78`. FU2 code is unchanged since `0d9a807`.
Checkpoints are not committed. `fu2_restore.zip` (kept by the user) holds the three deployed `A2.pt` files.

## Checks done on 2026-10-27 (local, read only)

- The six files are identical to the run's `nids_out/fu2/...` copies, and to those in `fu2_restore.zip` and its
  `RESTORE_MANIFEST.json` hashes. No test file exists in the run output.
- `summary.json` agrees with `rounds.json` in every seed:
  - the update status matches;
  - each attempt is accepted exactly when it has no gate reasons;
  - the relabel counts match.
- `final_sha256` equals the sha256 of the run's `poisoned_relabel/A2.pt`, and `A0_sha256` equals its `A0.pt`.
- `T_unchanged`, `source_run_unchanged` and all three P0 checks are true in every seed:
  - the buffer before E1 equals the source's;
  - E1 C1 has the same weights as the source C1;
  - E1 R1 has the same weights as the forced diagnostic.
- E1's validation numbers equal the committed sweep CSVs: C1 equals `decisions.csv` (poisoned_repair, E1, C1), and
  R1 equals `diagnostics.csv` (poisoned, E1, threeway+relabel). This covers Syn recall, mean p(Syn), benign FAR and
  n relabelled.

## Validation results

| seed | update | attempt | accepted | Syn | LDAP | UDP | benign FAR | relabelled / flipped stored Syn | gate reason |
|---|---|---|---|---|---|---|---|---|---|
| 42 | E1 | C1 | no | 0.0000 | 0.9869 | - | 0.0007 | - / 2000 | old Syn recall 0.000 < floor 0.945 |
| 42 | E1 | R1 | yes | 0.9953 | 0.9874 | - | 0.0003 | 1995 / 2000 | |
| 42 | E2 | C1 | no | 0.0000 | 0.9887 | 0.9980 | 0.0014 | - / 2000 | old Syn recall 0.000 < floor 0.945 |
| 42 | E2 | R1 | yes | 0.9953 | 0.9887 | 0.9985 | 0.0017 | 1995 / 2000 | |
| 43 | E1 | C1 | no | 0.0000 | 0.9852 | - | 0.0002 | - / 2000 | old Syn recall 0.000 < floor 0.946 |
| 43 | E1 | R1 | yes | 0.9941 | 0.9917 | - | 0.0007 | 1994 / 2000 | |
| 43 | E2 | C1 | no | 0.0000 | 0.9817 | 0.9993 | 0.0017 | - / 2000 | old Syn recall 0.000 < floor 0.944 |
| 43 | E2 | R1 | yes | 0.9955 | 0.9843 | 0.9993 | 0.0012 | 1994 / 2000 | |
| 44 | E1 | C1 | no | 0.0000 | 0.9891 | - | 0.0027 | - / 2000 | old Syn recall 0.000 < floor 0.946 |
| 44 | E1 | R1 | yes | 0.9961 | 0.9878 | - | 0.0019 | 1995 / 2000 | |
| 44 | E2 | C1 | no | 0.0000 | 0.9878 | 0.9989 | 0.0008 | - / 2000 | old Syn recall 0.000 < floor 0.946 |
| 44 | E2 | R1 | yes | 0.9960 | 0.9856 | 0.9989 | 0.0003 | 1995 / 2000 | |

Every relabelled row was relabelled to its true class (Syn). The 5-6 flipped rows per seed that were not relabelled
have weight 1 with no quarantine. Under the registered rule, that means the teacher itself gives them p(BENIGN) >= 0.9,
so they train as BENIGN.

## Predictions

- **H1 (E1: C1 rejected, R1 accepted, 3/3):** observed 3/3. **This is not independent evidence**: the P0 checks show
  E1 is the sweep diagnostic replayed.
- **H2 (E2: C1 rejected and R1 accepted in >= 2/3 seeds; the model knows BENIGN, Syn, LDAP, UDP):** observed 3/3
  on validation. In every seed, E2 C1 again drove Syn val recall to 0.000 (all stored Syn labels are still BENIGN),
  and the relabelling R1 restored it to 0.994-0.996.
- **H3 (test):** met in 3/3 seeds (see below).

No significance claim (3 seeds). The validation rows also decided the gate, so these numbers are not a held-out
estimate.

## Held-out test (scored once, 2026-10-27 13:53 UTC)

`fu2_test.zip` (sha256 `951d8eb4a5d4cd7dce189d34353f088ff7247c54e5177a0df023caf95921bae5`) came from the same Kaggle
session as validation (`RESTORE = False`). Its deployed models are byte-identical to that session's validation output.
Checks:
- The zip's `summary.json` and `rounds.json` are byte-identical to commit `db4ca21`.
- Each `final_test.json` records `decisions_commit` `db4ca21d0a7be3bcb8a0be48c157f0d64af8d229`, and each seed was
  scored once.
- The test rows (46,527, sha256 `09245b03...`) equal those of the source runs' clean, poisoned and poisoned_repair
  arms. The control numbers equal the committed `04_sweep/final_test.csv`.

`python make_fu2_tables.py --fu2 <fu2_test.zip>/fu2 --src <notebook 04 nids_out>` writes `test/`:
- the three `final_test.json` files, unchanged;
- `test_recall.csv`, `test_summary.csv` and `h3.csv`.

| seed | arm | BENIGN | Syn | LDAP | UDP | macro recall | benign FAR | Syn rows -> BENIGN |
|---|---|---|---|---|---|---|---|---|
| 42 | clean | 0.9982 | 0.9954 | 0.9886 | 0.9992 | 0.9953 | 0.0018 | 51 |
| 42 | poisoned | 1.0000 | 0.9951 | 0 | 0 | 0.4988 | 0 | 58 |
| 42 | poisoned_repair (soft) | 1.0000 | 0.9951 | 0 | 0 | 0.4988 | 0 | 58 |
| 42 | **poisoned_relabel** | 0.9985 | 0.9951 | 0.9878 | 0.9979 | 0.9948 | 0.0015 | 54 |
| 43 | clean | 0.9992 | 0.9954 | 0.9836 | 0.9994 | 0.9944 | 0.0008 | 52 |
| 43 | poisoned | 1.0000 | 0.9954 | 0 | 0 | 0.4988 | 0 | 55 |
| 43 | poisoned_repair (soft) | 1.0000 | 0.9954 | 0 | 0 | 0.4988 | 0 | 55 |
| 43 | **poisoned_relabel** | 0.9989 | 0.9951 | 0.9766 | 0.9994 | 0.9925 | 0.0011 | 55 |
| 44 | clean | 0.9995 | 0.9954 | 0.9832 | 0.9977 | 0.9939 | 0.0005 | 52 |
| 44 | poisoned | 1.0000 | 0.9954 | 0 | 0 | 0.4988 | 0 | 55 |
| 44 | poisoned_repair (soft) | 1.0000 | 0.9954 | 0 | 0 | 0.4988 | 0 | 55 |
| 44 | **poisoned_relabel** | 0.9993 | 0.9954 | 0.9832 | 0.9975 | 0.9939 | 0.0007 | 52 |

What the table shows:
- **Controls.** The control arms come from the sweep and were not rerun. `poisoned` and `poisoned_repair` deploy A0
  (BENIGN and Syn only), so LDAP and UDP are 0 by construction.
- **H3** (Syn >= 0.95, LDAP and UDP >= 0.80, benign FAR <= 0.05) is met in 3/3 seeds.
- **FU2 vs clean.** FU2 is within 0.007 of clean on LDAP and UDP and within 0.0003 on Syn in every seed, with no
  higher benign FAR.
- **Syn rows -> BENIGN** (post hoc, descriptive) is 52-55 for FU2 vs 51-52 for clean, so the residual poison effect
  on Syn is small.

What this does and does not show:
- **It shows** that, at f = 1.0 in this setting, deploying relabelling recovers a model on par with the clean arm,
  where the registered soft repair cannot.
- **It does not show** that relabelling was pre-registered as the method. It was chosen after the sweep diagnostic,
  and E1 is a replay.
- **Generality is untested:**
  - CII only, and one attacked class whose clean rows the teacher classifies at p >= 0.9.
  - The teacher knows only BENIGN and Syn.
  - Three seeds, and no other fraction was run under FU2.
