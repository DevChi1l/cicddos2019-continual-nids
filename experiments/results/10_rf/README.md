# RF (post hoc): repeated small Syn replay corruption under fixed-class traffic refresh

Pre-registration: `experiments/prereg_refresh.md` (commit `b254211`, before any RF code or run). Code: commit
`32e890a` (`s14_refresh.py`, notebook 10). Kaggle T4, notebook 10 cells 1-6, then cells 8-9 (test, once) in the same session; validation T4 time 20.8 min.

**Read this first.**
- **Setting.** RF tests **traffic refresh with a fixed class set, not new-class arrival.** E0 = BENIGN + Syn + TFTP,
  then six updates of fresh, disjoint BENIGN and TFTP rows. No class is new, and no fresh Syn rows arrive.
- **Post hoc.** The design and TFTP were chosen after RC (`08_rc/`) and RCS (`09_rcs/`) stopped at their clean
  stage.
- **Scope.** CII only, one attacked class, 3 seeds; no significance claim.
- **Status.** Validation committed in `b0d6b63`; held-out test scored once afterwards (section below).

## Files (unchanged from `rf_val.zip`, sha256 cfdc5307…; byte-checked)

All 67 files are committed unchanged:
- `rf/pilot.json`, `rf/summary.json`, `rf/trajectory_val.csv`, `rf/diag_val.csv`, `rf/schedule_ref6.json` and
  `rf/timing.json`;
- `ref6/cii/experience_manifest.json`;
- for each of the 6 runs `runs/cii/rf_ref6_r{0.10,0.05}_s{42,43,44}/`: `config.json`, `run_manifest.json`,
  `A0_metrics.json`, `<arm>/rounds.json`, `<arm>/val_done.json` and `poisoned/r1_diag.json`.

`score-test` checks 66 of them byte for byte against this commit. It does not check `timing.json`, which a restore
session appends to.

Checks made before committing:
- **Provenance.**
  - `rf_val.zip` equals the copy inside `results_10.zip` and the session's `nids_out` files (67/67).
  - The session's code is `32e890a`.
- **Strict clean pilot (`rf_clean.zip`, sha256 36f45406…).**
  - Its 18 files are byte-identical to the same files in `rf_val.zip`, except `timing.json`. Its calls are an exact
    prefix of the final `timing.json`.
  - The 12 `clean_files_sha256` in `pilot.json` match the committed files.
  - Recomputing the predicate from `rounds.json` reproduces `pilot.json`.
- **Registered settings.**
  - The schedule is `[[Syn, TFTP], [TFTP] x 6]`, and the protected val rows equal the recorded check (BENIGN 9,507,
    Syn 5,973, TFTP 5,991).
  - Every config has the registered settings: Syn -> BENIGN, poison-every-update, defer-test, soft R1, forced soft-R1
    diagnostic on the poisoned arm only, the gate (0.05 / 0.80 / 0.05 / 0.05; no confidence gate), 5 update epochs and
    2,000 rows per class.
- **Integrity.**
  - In every run the teacher is unchanged, there is no `A0_test`, and every arm has `test_scored: false`.
  - No update had a new class.
  - The flipped stored Syn rows before update t equal rho x 2,000 x t in both poisoned arms.
- **Trajectories and models.**
  - Every value in `trajectory_val.csv` (decision, Syn recall, mean p) matches `rounds.json`.
  - All 114 deployed-model hashes in `summary.json` (6 runs x 3 arms x A1..A6, plus 6 A0) match the
    `rf_restore.zip` manifest.

## Strict clean pilot (seeds 42-44): PROCEED

The pilot was stable in 3/3 seeds:
- all 6 updates were accepted in every seed;
- the smallest gate margin was 0.048, 0.049 and 0.042;
- the largest change in Syn recall from A0 was 0.0020, 0.0010 and 0.0077.

## Validation results

| rho | Arm | C1 accepted | R1 ran | Syn val recall, A0 -> A6 (s42 / s43 / s44) | Mean p(Syn | Syn val rows), A0 -> A6 |
|---|---|---|---|---|---|
| 0.10 | clean | 18/18 | 0 | 0.9953 / 0.9950 / 0.9953 -> 0.9946 / 0.9960 / 0.9960 | 0.993 -> 0.990 / 0.994 / 0.993 |
| 0.10 | poisoned | **18/18** | 0 | -> **0.9948 / 0.9948 / 0.9933** | 0.993 -> **0.756 / 0.786 / 0.770** |
| 0.10 | poisoned_repair | 18/18 | **0** | identical to poisoned (same weight hashes at every t) | identical to poisoned |
| 0.05 | poisoned | 18/18 | 0 | -> 0.9940 / 0.9953 / 0.9933 | 0.993 -> 0.908 / 0.910 / 0.889 |
| 0.05 | poisoned_repair | 18/18 | 0 | identical to poisoned | identical to poisoned |

- **The gate passed every poisoned update and repair never triggered.** At rho = 0.10 all 18 poisoned updates
  passed. After E6, 60% of the stored Syn rows carried the BENIGN label.
  - Syn val recall stayed near 0.994: 0.993-0.995 at A6, with a lowest dip of 0.984.
  - The smallest Syn gate margin of a poisoned C1 was 0.038.
- **Confidence did fall.** Syn mean p fell to about 0.770 (0.756-0.786), with Spearman correlations against the
  flipped share of -1.0, -0.96 and -0.96.
- **The other measures did not move.**
  - Benign FAR stayed at or below 0.0029.
  - TFTP recall stayed between 0.996 and 0.998.
  - The clean arm is identical across rho (determinism check).
- **The forced soft-R1 diagnostic shows what repair would have done.** It is logged only and never deployed.
  - It would have passed the gate at all 36 poisoned updates, and kept Syn mean p at 0.96-0.995.
  - The teacher gave the flipped rows a mean weight of 0.001-0.023, and the correct stored rows 0.993-0.995.
  - Repair can undo this corruption, but a gate that checks recall never asks for it.

## Registered predictions (validation)

| Prediction | Result | Evidence |
|---|---|---|
| **RF1 ratchet** (rho 0.10: C1 accepted at >= 5 of 6 updates in >= 2/3 seeds **and** drift > 0.05 in >= 2/3) | **not supported** | the first part held in 3/3 seeds (6/6 accepted); the drift was 0.0005, 0.0002 and 0.0020, so 0/3 seeds exceeded 0.05. Recall did not ratchet down |
| **RF2 silent confidence loss** (rho 0.10, 3/3 seeds: Spearman < 0 and mean p at A6 < A0) | **supported** | 3/3 seeds; Spearman -1.0 / -0.96 / -0.96; mean p 0.993 -> 0.756 / 0.786 / 0.770 |
| **RF3 repair is gated** (repair arm deploys the poisoned arm's models until its first C1 rejection) | **supported**, in the trivial form | no C1 rejection in any run, so R1 never ran and the repair arm's 36 deployed models (A1..A6, 6 runs) have the poisoned arm's weight hashes |
| Clean control (not a prediction) | holds | the pilot was 3/3 stable; at rho 0.05 the clean arm is identical to rho 0.10's |

`summary.json` reports these as `RC1`-`RC4`; that is s12's naming, and RC4 is the clean control.

## Held-out test (scored once at decisions commit `b0d6b63`)

- **How it was scored.** In the same Kaggle session, after the validation commit, `s14_refresh.py score-test`
  scored every arm's deployed A0..A6 of all 6 runs: 126 models, 2026-10-27 20:22-20:24 UTC.
- **Files.** The 19 files of `rf_test.zip` (sha256 9a52a128…) are committed unchanged in `test/rf/test/`.
  `make_rf_tables.py` writes `test/test_final.csv` (A6 per seed) and `test/test_summary.csv` (mean / min / max over
  seeds), reading saved files only.
- **Checks:**
  - the session's 66 validation files equal commit `b0d6b63`, and `rf_val.zip` and `rf_restore.zip` are unchanged;
  - all 18 test files record `decisions_commit` `b0d6b63`;
  - all 126 scored checkpoints have the sha256 recorded in the committed `summary.json`;
  - every model was scored on the same rows: 42,962 rows (BENIGN 19,028, Syn 11,946, TFTP 11,988), one row hash;
  - `test_trajectory.csv` equals the JSON files;
  - the clean arm is identical across rho, and `poisoned_repair` is identical to `poisoned`.

**Deployed model after E6 (A6), test, mean over seeds 42-44 (min-max):**

| Arm | Stored Syn labels flipped | Syn recall | Syn mean p | TFTP recall | Benign FAR | Macro recall (3 classes) |
|---|---|---|---|---|---|---|
| clean | 0% | 0.9952 (0.9949-0.9954) | **0.9922** (0.9904-0.9933) | 0.9974 | 0.0007 (0.0002-0.0017) | 0.9973 |
| poisoned, rho 0.05 | 30% | 0.9943 (0.9930-0.9951) | 0.9027 (0.8889-0.9104) | 0.9976 | 0.0007 (0-0.0011) | 0.9971 |
| poisoned, rho 0.10 | **60%** | **0.9943** (0.9936-0.9950) | **0.7707** (0.7565-0.7859) | 0.9974 | 0.0007 (0-0.0011) | 0.9970 |
| poisoned_repair, rho 0.05 / 0.10 | 30% / 60% | identical to poisoned | identical to poisoned | | | |

A0, before any update and the same for every arm: Syn recall 0.9951, Syn mean p 0.9928.

- **Recall did not drop.** At 10% per update, the flips accumulated to 60% of the stored Syn replay labels, and test
  Syn recall stayed at about 0.9943 (clean 0.9952).
- **Confidence did.** Mean Syn confidence fell to 0.7707, against 0.9922 for the clean arm.
- **The gate never reacted.** No update was rejected and no repair was deployed. The repair arm's deployed models
  are the poisoned arm's, so the confidence loss is in the repair arm too.
- **Test agrees with validation** (val A6: Syn recall 0.993-0.995, mean p 0.756-0.786).

## What this does and does not show

- **What it shows.** Under fixed-class refresh, repeated 5-10% Syn->BENIGN replay corruption, up to 60% of the
  stored Syn rows:
  - did **not** lower Syn recall over 6 updates (validation and test);
  - **did** steadily lower the model's confidence on Syn (test mean p 0.7707 vs 0.9922 clean);
  - was never seen by the recall-based gate, so no update was rejected and no repair was deployed.
- **RF did not repair the confidence loss.** The forced soft-R1 diagnostic kept validation confidence near 0.99, but
  it was never deployed and never scored on test.
- **What it does not show.** It is not evidence about new-class arrival, other attacked classes, CI, more updates or
  larger shares. Nor is it evidence that all 12 classes were learned continually: RF uses 3 classes.
- **Not a design change.** s3 already has a confidence gate (`--gate-conf-drop`), but it was not registered here. RF
  says nothing about how that gate would behave.
