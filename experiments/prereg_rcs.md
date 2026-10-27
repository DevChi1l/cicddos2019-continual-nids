# Pre-registration: repeated small replay corruption on a stability-screened schedule (RCS)

Written 2026-10-28, **before any RCS code or run**. The git commit that adds this file is its timestamp. Changes
after it go under "Deviations" with a date and a reason.

## Relation to RC (notebook 08)

RC (`prereg_repeated_corruption.md`) **stopped at its clean pilot**; see `results/08_rc/`. Its 6-update schedule
was not stable without poison: SNMP was rejected in 3/3 seeds and MSSQL in 2/3, every time because old-class LDAP
recall fell.

RCS is a **new, separate experiment**:
- new Stage 1b tags (`rcs6`, `rcs5`), run names `rcs_<tag>_r<rho>_s<seed>`, work folder `rcs/`, and results in
  `results/09_rcs/`;
- code `s13_rcs.py` and notebook `09_repeated_corruption_stable.ipynb`;
- new seeds for every run that carries poison.

No RC file is an RCS result, and no RCS file is an RC result. RC's pilot outcome (clean arm, validation only) was
seen before this design, and it is why the candidates below look the way they do (disclosed).

## Question (unchanged from RC)

The gate's floor for an old class is relative to the **last accepted** model (old recall >= ref - 0.05). Can small
corruptions that each stay inside that allowance be accepted one after another, so that Syn drifts down over many
updates without a rejection, and so without repair ever running?

## What RCS changes, and only this

1. **The schedule** is chosen by a clean-only, validation-only screen of two candidates fixed below. It is then
   **confirmed on fresh seeds before any poisoned run**.
2. **Stability is stricter than RC's pilot.** It must hold in 3/3 seeds (RC needed 2/3), and adds a gate-margin and
   a Syn-band condition.
3. **Seeds.**
   - Screening uses 42, 43 and 44 (selection only).
   - Confirmation and every poisoned run use 45, 46 and 47.

Everything else is RC's design, unchanged:
- the capped 12-label Stage 1 rows, `--attack-cap 9312`, CII only;
- E0 = BENIGN + Syn, and the frozen teacher T = E0;
- model, recipe, 5 update epochs;
- the gate (old >= ref - 0.05, new >= 0.80, benign FAR <= 0.05, new-class FAR <= 0.05);
- 2,000 buffer rows per class per accepted update;
- the attack: rho x 2,000 more stored Syn rows flipped to BENIGN before **every** update, with flips that persist,
  at rho = 0.10 then 0.05;
- the arms (clean / poisoned / poisoned_repair with the registered soft R1) and the forced soft-R1 diagnostic on the
  poisoned arm;
- the fixed horizon, the 90-min T4 cap and the invariants;
- RC's "Clarifications before execution" 1, 3, 4, 5, 6, 7, 9 and 10, with T updates in place of 6.

## Candidate schedules (priority order; no others)

| Priority | Tag | E1 .. ET | T | Reason (from RC's clean pilot, validation) |
|---|---|---|---|---|
| 1 | `rcs6` | MSSQL, NTP, SNMP, SSDP, TFTP, **LDAP** | 6 | RC's six classes, with LDAP moved last. In RC every rejection was LDAP being forgotten when SNMP or MSSQL came after it. **Risk:** the overlap works both ways, so learning LDAP last may lower SNMP or MSSQL recall instead |
| 2 | `rcs5` | MSSQL, NTP, SNMP, SSDP, TFTP | 5 | LDAP dropped; the other five keep name order |

- **Minimum horizon:** 5 updates (as RC). Both candidates meet it. There is no third candidate, no 4-update fallback
  and no second screen.
- **Benign.** CII benign is split into T + 1 disjoint portions:
  - 8,166-8,167 rows each for `rcs6`;
  - 9,527-9,528 rows each for `rcs5`.

  So E0 (and T) differ between the candidates.
- **Feasibility (2026-10-28, before registration).** Stage 1b was run on a local copy of the real Stage 1 rows; no
  training was run. Both schedules are accepted. The protected val rows kept for scoring are:

  | | BENIGN | Syn | MSSQL | NTP | SNMP | SSDP | TFTP | LDAP |
  |---|---|---|---|---|---|---|---|---|
  | `rcs6` | 9,507 | 5,973 | 5,591 | 5,994 | 4,955 | 4,719 | 5,993 | 2,301 |
  | `rcs5` | 9,507 | 5,973 | 5,591 | 5,994 | 4,959 | 4,719 | 5,993 | - |

  `s13_rcs.py prepare` must reproduce these counts, or it refuses.

## Stability predicate S (one clean run, clean arm only, validation)

S holds for a run if all three conditions hold:
- **(a)** C1 is accepted at every update t = 1..T.
- **(b)** At every C1, the smallest gate margin is >= 0.01. The margins are as in RC clarification 4:
  - for each old class except BENIGN: recall - (ref - 0.05);
  - for the new class: recall - 0.80;
  - 0.05 - benign FAR;
  - 0.05 - new-class FAR.

  Why: in RC, seed 42 accepted E5 with a margin of +0.000. If the clean control passes an update only on a knife
  edge, a poisoned-arm rejection at that update could come from training noise in another class, not from the poison.
- **(c)** max over t of |Syn val recall(A_t) - Syn val recall(A0)| <= 0.02 (RC4's band).

## Stopping rules (fixed before execution)

1. **Stage S, the screen.**
   - Both candidates are run: clean arm only, seeds 42-44, validation only.
   - The selected schedule is the first candidate, in priority order, for which S holds in **3/3** seeds.
   - If no candidate qualifies, **STOP**: RCS ends and `rcs/screen.json` is reported.
2. **Stage C, the confirmation.**
   - Only the selected schedule is run: clean arm only, seeds 45-47, inside the rho = 0.10 run folders.
   - **Proceed only if S holds in 3/3 seeds.** Otherwise **STOP** and report.
   - `rcs/go.json` records the decision, its UTC time and the sha256 of the confirmation runs' clean files.
   - No poisoned arm can start without `go.json` saying proceed; `s13 run` refuses.
3. **Stage P, the poisoned runs.**
   - rho = 0.10 runs first. They resume the confirmation runs, so their clean arm is the control and is not
     retrained. `summarize` asserts that its files are byte-identical to the hashes in `go.json`.
   - The rho = 0.05 runs train their own clean arm (a determinism check, as in RC).
   - If the measured T4 time, including Stages S and C, exceeds 90 min after rho = 0.10, stop and report.
4. **Fixed horizon T.** No early stop when Syn collapses (that is the result), and no extension.
5. **Invariants.** A changed teacher hash, an input-hash mismatch or a test-row hash mismatch means stop and
   report. A crashed run resumes with s3 `--resume`.
6. **No changes.** Rho, target, candidates, predicate, gate, recipe and repair method do not change after seeing any
   result of Stage S, C or P.

## Predictions (low confidence; seeds 45-47, horizon T)

- **RCS1 (ratchet).** Two parts, at rho = 0.10:
  - the poisoned arm accepts C1 at >= T - 1 updates in >= 2 of 3 seeds;
  - its drift (Syn val recall of A0 minus A_T) is > 0.05 in >= 2 of 3 seeds.

  At rho = 0.05 there is no prediction about direction.
- **RCS2 (silent confidence loss).** At rho = 0.10, in 3/3 seeds, the poisoned arm's mean p(Syn | Syn val rows):
  - has a Spearman correlation < 0 with the cumulative flipped share;
  - is lower at A_T than at A0.
- **RCS3 (repair is gated).** In every run, up to the repair arm's first C1 rejection, the repair arm deploys models
  with the same weight hashes as the poisoned arm.
- **Clean control (RC4).** At rho = 0.10 this holds by construction (Stage C), so it is **not** a prediction. It is
  reported for rho = 0.05 as a determinism check.

## Test

Test is RC's procedure:
- only after the validation files are committed;
- scored once;
- every arm's deployed A0..A_T of the 6 main runs, on the protected test rows (a descriptive test trajectory).

No decision uses test. Screening runs are never scored on test.

## Reporting

- Per seed, the trajectory of every arm, with counts across seeds and no significance claims.
- Disclose:
  - the candidates were built after seeing RC's clean validation pilot (and the earlier all12 / FU1 tables, test
    included);
  - CII only, one attacked class, 3 seeds.
- If RCS stops at Stage S or C, report that no stable schedule of >= 5 updates was found under this gate and
  recipe; that is the result.

## Runtime estimate

Measured in RC's pilot: about 60 s per clean run on a T4 (E0 about 14 s, about 5 s per attempt), and about 20 s per
Stage 1b tag.

| Stage | Work | Time |
|---|---|---|
| prepare | 2 tags | about 1 min |
| S | 6 clean runs | about 6 min |
| C | 3 clean runs | about 3 min |
| P, rho 0.10 | 3 runs, 2 more arms each | about 6-10 min |
| P, rho 0.05 | 3 runs, 3 arms each | about 9-12 min |
| summarize | | about 1 min |

**Total: about 25-35 min of T4**, against the 90 min cap.

## Deviations

None.

## Outcome (added 2026-10-28; the design above is unchanged)

Stopped by stopping rule 1 at Stage S (Kaggle, code `c33e961`): `SELECTED = None`, with both candidates stable in
0/3 screening seeds. No confirmation, poisoned run or test was performed. Files: `results/09_rcs/`. The predictions
remain untested. Per the user's instruction, no further schedule search follows; any next experiment is a new
registration.
