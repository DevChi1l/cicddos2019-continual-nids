# FU2 validation results (notebook 06): deployed teacher relabelling at f = 1.0, seeds 42-44

Pre-registration: `experiments/prereg_followup_fu2.md` (commit `0d9a807`). **Validation only. Test is not scored.**
It is scored once by `s10_relabel.py score-test`, with `--decisions-commit` set to the commit that adds these files.

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
- **H3 (test):** not scored yet.

No significance claim (3 seeds). The validation rows also decided the gate, so these numbers are not a held-out
estimate.
