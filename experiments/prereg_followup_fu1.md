# Pre-registration: follow-up FU1 (retrying rejected clean blocks of the 12-label runs)

Written 2026-10-21, **before any FU1 run**. The git commit that adds this file is its timestamp. Changes after
that commit go under "Deviations" with a date and a reason. FU1 is a separate experiment: it never writes into the
`main`, sweep or all12 runs, and it never changes the gate or its thresholds.

## Why (from saved files, `experiments/results/04b_all12/`)

In `all12_cii_s42` and `all12_ci_s42`, the **clean** arm rejected these blocks on validation:
- **E1 (DNS, LDAP), both scenarios:** new DNS recall 0.677 (CII) / 0.580 (CI) < 0.80. Of the 4,367 DNS val rows,
  1,354 (CII) / 1,779 (CI) were predicted LDAP. LDAP itself reached 0.92 / 0.97.
- **E5 (UDP, UDPLag), both scenarios:** old SSDP recall 0.001 / 0.002 against a floor of about 0.94. 4,658 / 4,664
  of 4,718 SSDP val rows were predicted UDP. New-class FAR was 0.13 / 0.11.
- **E3, CII only:** old MSSQL recall 0.9390 against a floor of 0.9396.

s3 (`run_arm`) then keeps the previous model and moves on to block t+1. A rejected block's rows
(`ctx["exp"][t]`) are never used again and never enter the buffer. Its classes never join `A_cls`, so later gates
do not check them, and the final test scores them as recall 0. This follows the original rule "at most two attempts
per block (C1, R1)"; it is not a coding error.

Stage 1 dropped many label-conflict vectors for exactly these classes: UDP 191,822, SSDP 190,604, DNS 29,385,
LDAP 14,994 (`split_manifest.json`, `conflict_distinct_by_label`; the conflicting partner class was not recorded).
Our inference: SSDP/UDP and DNS/LDAP overlap heavily in this feature space, so a training change may not fix them.

## What is fixed

| Item | Value |
|---|---|
| Source runs (read only) | `runs/{cii,ci}/all12_{cii,ci}_s42`, clean arm only (start model `A_{t-1}`, `buffer_before_E{t}.csv`) |
| Inputs | must hash the same Stage 1/1b files as the source run's `config.json` (s9 refuses otherwise) |
| Gate | unchanged: old recall >= ref - 0.05, new >= 0.80, benign FAR <= 0.05, new-class FAR <= 0.05 |
| Teacher T | not used; `T.pt` file hash recorded and checked unchanged |
| Seeds | 42, 43, 44 for the update's training randomness (`torch.manual_seed(seed + 10 t)`, batch order `seed + t`, as s3). The start model and buffer stay those of the seed-42 run |
| Decisions | validation only; test is scored by `score-test` once, after the validation files are committed |

## Steps and recipes (one factor changed per recipe, relative to `base` = s3's C1)

| Step | Block | Recipes | Cells |
|---|---|---|---|
| 0 separability | - | 1-nearest-neighbour recall for DNS:LDAP and SSDP:UDP (training rows vs val rows); **descriptive, decides nothing** | 2 scenarios |
| 1 probe | E1 | base, ep10 (10 epochs), ep20 (20 epochs), lr3e-4, balanced (class-weight power 1.0) | 2 scenarios x 3 seeds |
| 2 probe | E5 | base, balanced (old replayed classes get the same total weight as new ones), distill (1.0 x 2^2 x KL(last accepted model, T=2) on old logits) | 2 scenarios x 3 seeds |

The recipe values were chosen after seeing the all12 validation confusions and the all12 test tables (commit
`c84d51a`). They are generic, untuned settings, and none of them uses a test number.

## Selection rule (`s9_followup.py decide`)

There are two failure types: new-class (probe E1) and old-class (probe E5).
- **Eligible:** a non-base recipe that passes the gate in >= ceil(2/3) of its cells, i.e. >= 4 of 6.
- **Choice:** the eligible recipe with the most passes; ties go to the highest median gate margin, then to the
  earlier recipe in the fixed list.
- **None eligible:** no recipe for that failure type.

Gate margin = the smallest of: old recall - floor, new recall - 0.80, 0.05 - benign FAR, 0.05 - new-class FAR.

## Retry policy (`s9_followup.py policy`, clean arm, whole schedule, both scenarios)

- **C1:** every block first gets the base update, exactly as in s3.
- **C2:** if C1 is rejected, one retry with the recipe selected for its failure type. An old class below its floor,
  benign FAR or new-class FAR counts as old-class; a new class below 0.80 counts as new-class. Old-class takes
  priority. With no selected recipe there is no C2.
- **D1:** a block still rejected is retried once more (recipe for its last failure type, else base) from the latest
  accepted model and buffer. This happens right after the next accepted block, or at the end of the schedule.
- **Then final:** the block's classes are reported as not learned.
- **Constraints:** at most 3 attempts per block, gate unchanged, and the buffer gains a block's rows only on
  acceptance.
- **Caveat:** each extra attempt reuses the same validation rows, so the final validation numbers are optimistic.
  The test score is the honest check.

## Predictions written down in advance (low confidence)

- **P0.** `base` with seed 42 reproduces the logged s3 C1 validation recalls exactly in all 4 (scenario, block)
  cells. Locally this already holds for the start models' reference metrics.
- **P1 (E1).** No E1 recipe reaches 4 of 6 passes. DNS recall stays below 0.80 because of the DNS/LDAP overlap.
- **P2 (E5).** No E5 recipe reaches 4 of 6 passes. `distill` raises SSDP recall but costs UDP recall and/or
  new-class FAR; `balanced` changes little. SSDP and UDP cannot both meet their thresholds.
- **P3.** If P1 and P2 hold, the policy runs only C1 + D1 (base). The same blocks stay rejected, except where a
  later start model or a larger buffer changes the outcome (descriptive).

These would be negative results. We report them as such: they mean the gate rejects these blocks because the
classes overlap, not because the training recipe is weak.

## Reporting

- Probe results are reported per cell (pass/fail and margin), with no significance claims.
- Step 0 is labelled as a ceiling hint.
- The policy runs are reported as validation decisions first. After the commit, they are compared with the s3
  clean arm on the same test rows (`score-test` checks the row hash).
- Not in scope: poisoned arms, threshold changes, per-class acceptance (accepting LDAP without DNS), and mild
  repeated corruption.

## Deviations

- **2026-10-21, before any FU1 run (procedure only):** added a resumable validation-to-test handoff.
  - After `policy`, `bundle` writes `fu1_restore.zip` (the validated `final.pt` per run, `summary.json`,
    `rounds.json`, `decisions.json` and a sha256 manifest; no test). `restore` checks every hash before copying it
    back.
  - `summary.json` now records `final.pt`'s sha256.
  - `score-test` refuses unless `decisions.json`, `summary.json` and `rounds.json` are byte-identical to their
    copies under `experiments/results/05_fu1/` in the named commit, and `final.pt` matches the recorded hash.
  - `policy` accepts only `<work>/followup/decisions.json`.
  - Recipes, selection rule, retry policy and gate are unchanged.
- **2026-10-27, after test was scored (descriptive step and packaging only; no decision, recipe, gate or result
  changed):**
  - Step 0 (separability) was a measurement artifact: float32 `torch.cdist` rounded small distances to 0. Its
    `separability.csv` is marked invalid in `results/05_fu1/README.md` and kept unchanged. The code now uses exact
    float64 distances and writes `separability_exact.csv`; it was rerun locally on train vs val rows only
    (`results/05_fu1/separability_exact_local/`). This happened after the test, so it supports no prediction.
  - Notebook 05 cell 9 (`fu1_test.zip`) needed a helper from cell 6, which a restore session skips; in the test
    session the zip was made by an ad-hoc cell with the same filter. The helper now lives in cell 1.
