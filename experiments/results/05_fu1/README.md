# FU1: validation outcome (code `8986446`) and test result (scored once, 2026-10-27)

**Conclusion and test numbers:** see [Test](#test-scored-once-after-the-validation-commit) and
[Conclusion](#conclusion) below. Validation files: `followup/` (unchanged). Test files: `test/`.

`followup/` is the content of `fu1_val.zip`, committed **unchanged**: 59 JSON/CSV files, no checkpoint, no test file.
Pre-registration: [`../../prereg_followup_fu1.md`](../../prereg_followup_fu1.md) (`66c88d4`, handoff `8986446`).
Nothing in `followup/` is a test result. Test was scored once, later, by `s9_followup.py score-test` with the sha of the commit
that added `followup/` (`5307b58`).
That command checks that `followup/decisions.json` and each run's `summary.json` / `rounds.json` are byte-identical
to these files.

| File | sha256 |
|---|---|
| `fu1_val.zip` (source of `followup/`) | `e25c0061a604ed87995eda0ab13d3f9d38580e40d4bee2b7007f13a92c823e9d` |
| `fu1_restore.zip` (not committed: holds `final.pt`; manifest created 2026-10-21 12:56 UTC) | `5d5d7cbb4000f47c8bb10ecb04520ac51124e67b2b5c04c12c305be3911b1a01` |

Checks before this commit:
- the zip lists exactly the JSON/CSV files of the session's `nids_out/followup/`, byte for byte;
- the only test-related key is the input hash of `clean_test.parquet`, and no `final_test.json` exists anywhere;
- `fu1_restore.zip` passes `s9_followup.py restore`, and its `decisions.json`, `summary.json` and `rounds.json` are
  byte-identical to these files;
- its `final.pt` equals the session's file and the sha256 recorded in `summary.json`.

## Validation outcome (decisions only)

- **P0 holds.** `base` with seed 42 reproduced the logged s3 C1 exactly in all 4 (scenario, block) cells
  (`reproduces_logged_C1` = 0.0).
- **E1 probe (new-class failure):** `ep20` passed the gate in 5 of 6 cells (median margin +0.023), so it was
  selected. `ep10` passed 1 of 6; base, lr3e-4 and balanced passed 0 of 6. DNS val recall with ep20 was
  0.80-0.86, LDAP 0.82-0.88. Prediction P1 ("no E1 recipe eligible") is **not supported**.
- **E5 probe (old-class failure):** no recipe passed (0 of 6 each), so no recipe was selected; P2 holds.
  `balanced` moves SSDP val recall up (0.11-0.98) only by giving up UDP (0.006-0.93), and it never meets both.
  `distill` changed almost nothing (SSDP <= 0.04).
- **Retry policy (both scenarios, identical block outcome):**

  | Block | Outcome |
  |---|---|
  | E1 (DNS, LDAP) | accepted at C2 (ep20) |
  | E2 (MSSQL, NTP) | rejected at C1 and D1 |
  | E3 (NetBIOS, SNMP) | rejected at C1 and D1 |
  | E4 (SSDP, TFTP) | accepted at C1 |
  | E5 (UDP, UDPLag) | rejected at C1 and D1 |

  E2 and E3 were accepted (CI) or partly accepted (CII E2) by s3. Under FU1 they now fail because DNS, learned just
  above 0.80, drops below its own floor (old DNS 0.49-0.68 vs floor about 0.77). At E3, LDAP and new-class FAR also
  fail.
- **The deployed models know 6 of 12 labels:** BENIGN, Syn, DNS, LDAP, SSDP, TFTP. By validation decisions, the
  s3 clean arm ended with 6 (CII: BENIGN, Syn, MSSQL, NTP, SSDP, TFTP) and 8 (CI) labels. FU1 swaps MSSQL/NTP (and
  CI's NetBIOS/SNMP) for DNS/LDAP; it does not add labels.
- **Policy weakness found:** the end-of-schedule D1 for E5 starts from the same model and buffer, with the same
  recipe and seed, as its C1. So it repeats C1 exactly (identical metrics) and is not a real second chance.

## INVALID: `followup/{cii,ci}/separability.csv` (float32; descriptive only, it decided nothing)

Its `share_identical_input` (about 0.6-0.93) and 1-NN recalls are a **measurement artifact**. `torch.cdist` in
float32 uses the matrix-product formula; with the protocol column scaled x100 (squared norms about 1e5), small
distances round to 0. Ties then go to whichever class comes first in the training index.

Local re-check on CPU, CII validation rows, exact float64 distances (no training, no test):
- no validation row has an identical training input (min distance > 0), as Stage 1b guarantees;
- 1-NN recall is SSDP 0.58 / UDP 0.55 (near chance, i.e. strong overlap) and DNS 0.84 / LDAP 0.85.

The file is kept unchanged as the record of what ran. **Do not cite its numbers.**

From the commit that adds `test/`, `s9_followup.py separability` computes exact float64 distances and writes
`separability_exact.csv` (column `distance = float64_exact`), so a future run can never produce or be confused with
the invalid file. That code was run locally on CPU on 2026-10-27, after test scoring, on the same notebook-04
inputs (train vs val rows only; no training, no test): `separability_exact_local/{cii,ci}.csv`.

| Scenario | DNS | LDAP | SSDP | UDP | identical inputs |
|---|---|---|---|---|---|
| CII | 0.843 | 0.850 | 0.578 | 0.555 | 0 |
| CI | 0.839 | 0.851 | 0.580 | 0.555 | 0 |

1-NN recall in a pair-only problem. SSDP/UDP is near chance (0.5), which fits the finding that no E5 recipe passed.
DNS/LDAP sits at about 0.84, close to the test recall the FU1 model reached (0.83). This is a descriptive hint, not
a ceiling proof (a model can beat 1-NN), and it was computed after the test, so it supports no prediction.

## Test (scored once, after the validation commit)

Provenance (`test/notebook05_test_session.ipynb`, the executed notebook downloaded from Kaggle):
- `fu1_test.zip` sha256 `eadb7523a27b8599e85b6ac8ceec33e5af65bee67567f563240f0b60ca80c310`; 7 files. Its
  `decisions.json`, `summary.json` and `rounds.json` are byte-identical to `followup/`. Only the two
  `final_test.json` are new; they are in `test/followup/`, unchanged.
- New Kaggle session, 2026-10-27, code `5307b58`: cell 1, then restore of `fu1_restore.zip` (11:28 UTC, every
  sha256 matched), then `score-test --decisions-commit 5307b58...` (CII 11:29:56, CI 11:30:04 UTC). The validation
  commit `5307b58` was made 2026-10-21 13:14 UTC, so test was scored after it. `score-test` checked the committed
  files and `final.pt`'s hash before scoring, and each `final_test.json` records `decisions_commit`.
- Same test rows as the all12 runs: 123,281 rows, rows sha256 `1eeeed6f...cef87` in both scenarios, equal to the
  s3 clean arm's (`make_fu1_tables.py` asserts this and that the s3 numbers equal `../04b_all12/final_test.csv`).
- Cell 9 (`fu1_test.zip`) failed in that session (`NameError`): it needed a helper defined in cell 6, which a
  restore session skips. The zip was made by an ad-hoc cell with the same file filter (last cell of the notebook).
  Fixed afterwards: the helper now lives in cell 1.
- The notebook file also shows cells 2-6 with outputs dated 2026-10-21 13:29 UTC. They come from the validation
  session (the cell timestamps say so), after `fu1_restore.zip` was made (12:56 UTC): probes were skipped as done,
  `decide` printed the same table, `policy` refused to overwrite. Nothing from them reached the restored files,
  which `score-test` checked against `5307b58`. No validation step ran in the test session.

Tables (`make_fu1_tables.py`): `test/test_recall.csv` (per class), `test/test_summary.csv`, `test/test_unlearned.csv`.

| Test (held out) | CII s3 clean | CII FU1 | CI s3 clean | CI FU1 |
|---|---|---|---|---|
| Labels learned (of 12) | 6 | 6 | 8 | 6 |
| DNS / LDAP recall | 0 / 0 | **0.831 / 0.834** | 0 / 0 | **0.837 / 0.829** |
| MSSQL / NTP recall | 0.970 / 0.956 | **0 / 0** | 0.938 / 0.980 | **0 / 0** |
| NetBIOS / SNMP recall | 0 / 0 | 0 / 0 | 0.936 / 0.971 | **0 / 0** |
| SSDP / TFTP recall | 0.987 / 0.996 | 0.993 / 0.996 | 0.989 / 0.994 | 0.993 / 0.996 |
| UDP / UDPLag recall | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Macro recall (12 labels) | 0.492 | 0.471 | 0.648 | 0.470 |
| Benign FAR (19,028 benign rows) | 0.0010 | 0.0018 | 0.0290 | 0.0084 |
| Attack rows predicted as some attack (post hoc) | 0.984 | 0.992 | 0.995 | 0.993 |

- DNS/LDAP test recall matches validation (0.80-0.86), so the repeated use of validation rows did not inflate it
  much. The remaining confusion is mutual: 1,352 / 1,323 of 8,738 DNS rows go to LDAP, 713 / 750 of 4,583 LDAP rows
  go to DNS (CII / CI).
- Classes FU1 did not learn are mostly flagged as *some* attack, but under the wrong label: MSSQL -> DNS (95-96%),
  SNMP -> LDAP (71-74%), UDP -> SSDP (>99%), UDPLag -> Syn (83%). NetBIOS is the exception: 16-18% of its rows are
  predicted BENIGN. The last row of the table is a post-hoc descriptive number, not a pre-registered metric.
- CI's lower benign FAR under FU1 is not an improvement of the method: in s3 CI, 524 of the 19,028 benign rows were
  predicted NetBIOS, and FU1 has no NetBIOS output.

## Conclusion

- **FU1 reached its narrow goal and lost on the overall one.** The pre-registered E1 recipe (`ep20`) made DNS and
  LDAP learnable, and that held on test (about 0.83 each). But the deployed models still know only 6 of 12 labels,
  the same as s3 clean in CII and fewer than its 8 in CI, and macro recall dropped (CII 0.492 -> 0.471,
  CI 0.648 -> 0.470).
- **Why (validation `rounds.json`):** accepting DNS/LDAP at recall just above 0.80 sets their floors at about
  0.77-0.80. Updating on the next blocks drops old DNS to 0.49-0.68: in CII at E2, of 4,367 DNS val rows 962 went to
  LDAP and 753 to MSSQL (C1); at E3, 515 to LDAP, 542 to NetBIOS and 357 to SNMP. So the old-class gate rejects
  MSSQL/NTP and NetBIOS/SNMP, also on the deferred retry. In s3, where DNS/LDAP were never accepted, E2 passed in both
  scenarios and E3 in CI (CII E3 missed the MSSQL floor by 0.0006). Overlapping classes compete, and whichever is accepted first blocks the others: the schedule order, more
  than the recipe, decides which labels survive (one seed, so this is a reading, not a tested claim).
- **SSDP/UDP stays unsolved.** No recipe passed (0 of 6 each), and exact 1-NN recall is near chance for this pair.
- **Policy weakness:** the end-of-schedule D1 for E5 repeated C1 exactly (same model, buffer, recipe and seed), so
  it was not a real second chance. Not fixed, since that would need retraining.
- **Limits:** one start model and buffer (seed 42), seeds 42-44 only in the probes, one test scoring, no
  significance claim. Validation rows were reused for up to 3 attempts per block.
- **What it implies (our inference):** with these features and a fixed per-class gate, the 12-label protocol is
  limited by class overlap (DNS/LDAP, SSDP/UDP; also the Stage 1 label-conflict counts), not by the update recipe.
  Better candidates are per-class acceptance or merged labels for overlapping pairs; both are untested proposals.
