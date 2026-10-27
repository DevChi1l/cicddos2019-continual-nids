# Pre-registration: repeated small replay corruption under fixed-class traffic refresh (RF), POST HOC

Written 2026-10-28, **before any RF code or run**. The git commit that adds this file is its timestamp. Changes
after it go under "Deviations" with a date and a reason.

## What this is, and what it is not

- **A post-hoc study.** RC (`results/08_rc/`) and RCS (`results/09_rcs/`) both stopped at their clean stage. In 15
  clean runs, every rejection was an old non-Syn amplification class forgotten when an overlapping new class
  arrived; Syn never moved by more than 0.001. This design was chosen **after** seeing those clean validation
  results, and the earlier all12 / FU1 / oracle tables, test included.
- **It tests traffic refresh with a fixed class set, NOT new-class arrival.** After E0, no update adds a class. Each
  update brings fresh, disjoint rows of BENIGN and TFTP, which the model already knows. Its results say nothing
  about class-incremental learning, and RC's predictions stay untested in that setting.
- **It is separate from RC and RCS:**
  - Stage 1b tag `ref6`, run names `rf_ref6_r<rho>_s<seed>`, work folder `rf/`, results in `results/10_rf/`;
  - code `s14_refresh.py`, notebook `10_refresh_corruption.ipynb`.

  No RC or RCS file is an RF result, and no RF file is an RC or RCS result.
- **It does not touch the completed defence result** (sweep soft R1 for f <= 0.95, and FU2 relabelling at f = 1.0;
  see the box in `results/SUMMARY.md`).

## Question

Before every refresh update, the attacker flips a few more stored Syn labels in the replay buffer to BENIGN. The
gate's floor for Syn is relative to the **last accepted** model. Can those updates be accepted one after another
while Syn recall drifts down by more than one update's allowance, so that repair never runs?

## Design

| Item | Value |
|---|---|
| Data | the capped 12-label Stage 1 rows (`clean_*.parquet`, as all12 / RC); CII only |
| E0 | BENIGN + Syn + TFTP; the teacher T = the E0 model, frozen, hash-checked |
| Updates E1..E6 | each brings a fresh portion of BENIGN and of TFTP; **no new class; no fresh Syn rows ever** (Syn exists after E0 only in the replay buffer, where the attacker works, as in RC) |
| Rows | BENIGN: 7 disjoint portions, as in RC. Syn: 9,312 rows in E0 (`--attack-cap 9312`). TFTP: all 35,970 train rows (fewer than 7 x 9,312), split into 7 disjoint portions of about 5,140 |
| Why TFTP (post hoc) | it was accepted in all 9 RC/RCS clean runs in which it arrived, with recall 0.993-0.996; it gives a second, non-attacked old class, so a Syn-specific effect can be told apart from general drift |
| Model, recipe, gate, buffer | as RC: E0 recipe, 5 update epochs, 2,000 buffer rows per class per accepted update. Gate: old >= ref - 0.05 for Syn and TFTP, benign FAR <= 0.05. The new-class recall and new-class FAR conditions do not apply, because no class is new |
| Attack | as RC: target Syn, flipped to BENIGN, in stored labels only. Before every update t = 1..6, rho x 2,000 more of the stored Syn rows (100 or 200) are drawn from those not yet flipped. Flips persist: 30% (rho 0.05) or 60% (rho 0.10) after E6 |
| Arms | as RC: clean / poisoned (gate only) / poisoned_repair (registered soft R1), plus the forced soft-R1 diagnostic on the poisoned arm, which is never deployed |
| Seeds | 42, 43, 44 |

**Code needed** (written after this file):
- s1b: `--allow-repeat`, off by default. A class may appear in several experiences. Its rows, at most
  attack-cap x appearances, are permuted and split into disjoint portions.
- s3: an update's new classes are the scheduled classes the model does not know yet.
- Both are no-ops for every existing schedule: the smoke test must keep all 30 checkpoint hashes, and `main`,
  the sweep and all committed runs stay valid.

## Strict clean pilot (stopping rule 1)

- **What runs:** the clean arm only, seeds 42-44, validation only, inside the rho = 0.10 run folders. It is one
  registered schedule, with no candidates and no fallback.
- **Stable run.** All three must hold (RCS's predicate):
  - **(a)** C1 is accepted at all 6 updates.
  - **(b)** At every C1, the smallest gate margin is >= 0.01. The margins are:
    - old Syn and old TFTP: recall - (ref - 0.05);
    - 0.05 - benign FAR;
    - the new-class terms are absent, because no class is new.
  - **(c)** max over t of |Syn val recall(A_t) - Syn val recall(A0)| <= 0.02.
- **Proceed only if every seed (3/3) is stable.** If any seed fails, **STOP**:
  - no poisoned or repair arm, no test, and no other schedule;
  - the pilot's files are saved and committed as the result.

  `rf/pilot.json` records the decision and the sha256 of the pilot's clean files. `run` refuses without
  `proceed = true`.

## Further stopping rules

2. **rho = 0.10 first, then 0.05.**
   - The rho = 0.10 runs resume the pilot runs, so their clean arm is the control and is not retrained.
     `summarize` asserts that its files are unchanged.
   - The rho = 0.05 runs train their own clean arm, a determinism check.
   - If the measured T4 time, including the pilot, exceeds 90 min after rho = 0.10, stop.
3. **Fixed horizon of 6 updates.** No early stop when Syn collapses, and no extension.
4. **Invariants.** A changed teacher hash, an input-hash mismatch or a test-row hash mismatch means stop. A crashed
   run resumes with s3 `--resume`.
5. **No changes.** Rho, target, schedule, predicate, gate, recipe and repair method do not change after any pilot or
   poisoned result.

## Predictions (low confidence)

- **RF1 (ratchet).** Two parts, at rho = 0.10:
  - the poisoned arm accepts C1 at >= 5 of 6 updates in >= 2 of 3 seeds;
  - its drift (Syn val recall of A0 minus A6) is > 0.05 in >= 2 of 3 seeds.

  At rho = 0.05 there is no prediction about direction.
- **RF2 (silent confidence loss).** At rho = 0.10, in 3/3 seeds, the poisoned arm's mean p(Syn | Syn val rows):
  - has a Spearman correlation < 0 with the cumulative flipped share;
  - is lower at A6 than at A0.
- **RF3 (repair is gated).** In every run, up to the repair arm's first C1 rejection, the repair arm deploys models
  with the same weight hashes as the poisoned arm.
- **Clean control.** At rho = 0.10 it holds by construction (the pilot), so it is **not** a prediction. It is
  reported at rho = 0.05.

The operational details are RC's clarifications 1, 3, 4, 5, 6, 7, 9 and 10.

## Test

- Only after the validation files are committed in git, `score-test` runs **once**, refusing unless every
  validation file is byte-identical to the decisions commit.
- It scores every arm's deployed A0..A6 on the protected test rows, as a descriptive test trajectory.
- No decision uses test.

## Reporting

- Per seed, every arm's trajectory, with counts across seeds and no significance claims.
- Always state:
  - post hoc;
  - **fixed-class traffic refresh, not new-class arrival**;
  - CII only, one attacked class, 3 seeds;
  - TFTP chosen from earlier results.

## Runtime estimate (T4)

Training rows per update grow from about 21k to 39k as the buffer grows, so an attempt takes about 5-10 s.

| Stage | Time |
|---|---|
| prepare (Stage 1 copy + Stage 1b) | about 1 min |
| pilot: 3 clean runs | about 4-5 min |
| rho 0.10: 3 runs, 2 more arms each | about 10-15 min |
| rho 0.05: 3 runs, 3 arms each | about 12-18 min |
| summarize | about 1 min |

**Total: about 30-40 min**, under the 90 min cap. If the pilot stops, about 6 min.

## Deviations

None.

**Erratum (2026-10-28; the design is unaffected).** "What this is" above says "In 15 clean runs". The
committed `rounds.json` files give 9 clean runs (RC 3, RCS 6) with 13 rejected updates. The rest of that sentence
(every rejection was a non-Syn class; Syn never moved by more than 0.001) holds for those 9 runs.

## Outcome, validation (added 2026-10-28; the design above is unchanged)

Kaggle, code `32e890a`, cells 1-6:
- the strict clean pilot was stable in 3/3 seeds (PROCEED);
- all 6 runs finished in 20.8 min of T4, with no stop;
- files: `results/10_rf/`.

Predictions on validation:
- RF1 is not supported: accepted 6/6 in 3/3 seeds, but drift was at most 0.002.
- RF2 is supported in 3/3 seeds.
- RF3 is supported (no C1 rejection, so the repair arm is identical to the poisoned arm).

Test was scored once afterwards, at decisions commit `b0d6b63` (`results/10_rf/test/`). On A6, mean over seeds:
- clean: Syn recall 0.9952, mean p 0.9922;
- rho 0.05: Syn recall 0.9943, mean p 0.9027;
- rho 0.10: Syn recall 0.9943, mean p 0.7707.
