# Pre-registration: follow-up FU2 (deploy teacher relabelling after C1 rejection, f = 1.0)

Written 2026-10-27, **before any FU2 run**. The git commit that adds this file is its timestamp. Changes after
that commit go under "Deviations" with a date and a reason. FU2 is a separate experiment: it never writes into
`main`, the sweep, the all12 runs or FU1, and it never changes the teacher, splits, gate or thresholds.

## Provenance: why this is a post-hoc follow-up

From the committed sweep (`prereg_sweep.md`, `eb6729f`; results `results/04_sweep/`, `results/SUMMARY.md` A):
- At f = 1.0 every stored Syn label in the buffer is BENIGN. The registered repair (soft R1) failed 6 of 6 times
  (Syn val recall <= 0.0002). The fallback kept A0, so LDAP and UDP were never learned (test: 0 in 3 of 3 seeds).
- "threeway+relabel" was registered **only as a forced diagnostic** (`--forced-r1` on the clean and poisoned arms,
  H5 "no direction predicted"). Its 0.9 / 0.1 thresholds were chosen after seeing `main`'s teacher weights.
  It was never a decision, never deployed, never grew the buffer and never scored on test.
- At f = 1.0 that diagnostic "would pass" the gate in 6 of 6 cases (E1 and E2, seeds 42-44; validation only).
  1,994-1,995 of 2,000 flipped rows were relabelled, all to Syn. Caveat: in the poisoned arm E1 was rejected,
  so the E2 diagnostic started from A0 (BENIGN, Syn) and added UDP without LDAP.

So FU2's method and poison fraction were chosen **after** seeing that diagnostic. FU2's E1 replays the diagnostic
exactly (same start, buffer, seeds and code path) and is therefore **not independent evidence**. New evidence is:
(i) E2 after relabelling was deployed at E1, where the model knows LDAP, the buffer holds LDAP rows and the Syn flips
are still stored; (ii) the held-out test of the deployed model, scored once after this validation is committed.

## What is fixed

| Item | Value |
|---|---|
| Source runs (read only) | `runs/cii/sweep_f1.0_s{42,43,44}` (notebook 04, code `3e7807c`; in its output, the same input as notebook 05) |
| Inputs | must hash the same Stage 1/1b files as each source run's `config.json` (s10 refuses otherwise) |
| Teacher T | the source run's `T.pt`, loaded read only; its state sha256 must equal the run manifest's before and after |
| Start model | the source run's `A0.pt`, copied and hash-checked |
| Poison | Syn -> BENIGN, f = 1.0 of the stored Syn rows, once before E1 (s3's rule, seed + 2). Flips persist all run |
| New arm | `poisoned_relabel` = s3's `poisoned_repair` code path (`S3.run_arm`), R1 method threeway+relabel |
| R1 rule | p_T(stored) >= 0.9 -> weight 1; <= 0.1 -> weight 0; else 0.5. If p_T(stored) <= 0.1 and max p_T >= 0.9, the label used for training becomes T's prediction (weight 1). Classes T never saw keep label and weight 1. The stored buffer is never changed |
| Attempts per update | C1 (plain stored labels); if rejected, one R1 from the last accepted model. Accepted -> buffer gains the update's rows; rejected -> previous model kept |
| Gate, splits, hyperparameters, seeds | the source run's (`config.json` args), unchanged |
| Controls | the source runs' clean, poisoned and poisoned_repair (soft) arms, read only, not rerun |
| Test | `run` scores nothing. `score-test` scores once per seed, after the validation files are committed under `experiments/results/06_fu2/`; it checks those files byte for byte and the model's sha256 |

## Checks before any claim (P0)

- `buffer_before_E1.csv` is byte-identical to the source run's `poisoned_repair` one.
- E1 C1 has the same weights (state sha256) as the source run's `poisoned_repair/E1_C1.pt`.
- E1 R1 has the same weights as the source run's forced diagnostic `poisoned/E1_R1diag_threeway_relabel.pt`.

If a check fails on Kaggle (GPU nondeterminism), we report it and treat E1 as a new result, not a replay.

## Predictions (validation; 3 seeds, so counts only)

- **H1 (not independent, follows from P0 and the diagnostic).** E1: C1 rejected and R1 accepted in 3 of 3 seeds.
- **H2 (new).** E2: C1 is rejected (the stored Syn labels are still all BENIGN, so Syn is forgotten again) and R1 is
  accepted, in >= 2 of 3 seeds. The deployed model then knows BENIGN, Syn, LDAP and UDP.
- **H3 (test, scored once, descriptive).** Per seed, the deployed model reaches Syn recall >= 0.95, LDAP and UDP
  recall >= 0.80, benign FAR <= 0.05. It is shown next to the clean, poisoned and soft-repair arms on the same
  test rows. No significance claim.
- **Relabel accuracy (diagnostic).** n relabelled and n relabelled to the true label per R1, from the stored
  ground-truth column, which training never reads.

## Rules

- No other fraction, method, threshold or seed is run under FU2. A crash is handled by re-running the cell:
  finished seeds are skipped and an unfinished seed restarts from scratch (deterministic).
- If H2 fails, that is the result. No retuning.
- Reporting must say: relabelling was deployed in a follow-up chosen after seeing a forced diagnostic; f = 1.0
  only; one attacked class; T knows only BENIGN and Syn, so relabelling can repair only labels of those classes.

## Runtime

The sweep's T4 timings are about 9 s per attempt and at most 4 attempts per seed, so about 2 min of training.
With clone, data loading and checks: about 10 min of T4 time, about 15 min of wall time.

## Deviations

(none yet)
