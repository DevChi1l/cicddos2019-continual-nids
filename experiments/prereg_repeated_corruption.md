# Pre-registration: repeated small replay corruption over many updates (RC)

Written 2026-10-27, **before any RC code or run**. The git commit that adds this file is its timestamp. Changes
after it go under "Deviations" with a date and a reason.

**This is not the sweep.** The sweep (`prereg_sweep.md`) flipped a large fraction (75-100%) of stored Syn labels
**once**, before E1, over 2 updates. RC adds **new** small corruption (5% or 10%) **at every update** over 6
updates and follows the trajectory. No sweep number may be presented as an RC result, or the reverse.

## Question

The gate's floor for an old class is relative to the **last accepted** model (old recall >= ref - 0.05). Can
small corruptions that each stay inside that allowance be accepted one after another, so that the attacked class
drifts down over many updates without any update ever being rejected (and so without repair ever running)? What
do the clean, poisoned (gate only) and repair arms look like over time?

## Design

| Item | Value |
|---|---|
| Data | the capped 12-label Stage 1 rows (`clean_*.parquet`, as all12); new Stage 1b tag `rep6`, scenario CII only |
| Schedule | E0 = BENIGN + Syn; then one class per update in name order: E1 LDAP, E2 MSSQL, E3 NTP, E4 SNMP, E5 SSDP, E6 TFTP. 6 updates, 9,312 attack rows per class (`--attack-cap 9312`), CII benign in 7 disjoint portions of 8,166-8,167 rows |
| Class choice (disclosed, post hoc) | DNS, UDP, UDPLag and NetBIOS are left out because of the committed all12 / FU1 results (incl. test tables seen): DNS/LDAP and SSDP/UDP overlap; UDPLag rows are predicted Syn (the attacked class); NetBIOS and BENIGN are confused both ways (CI s3: 524 benign test rows -> NetBIOS; FU1: 16-18% of NetBIOS -> BENIGN). Goal: a schedule the clean arm can learn, so a trajectory change is due to the poison |
| Feasibility checked 2026-10-27 | Stage 1b on a scratch copy accepts this schedule; protected val rows: BENIGN 9,507, Syn 5,973, LDAP 2,297, MSSQL 5,592, NTP 5,994, SNMP 4,961, SSDP 4,719, TFTP 5,994. No training was run |
| Model, gate, buffer | as all12 / sweep: E0 recipe, 5 update epochs, gate (old >= ref - 0.05, new >= 0.80, benign FAR <= 0.05, new-class FAR <= 0.05), 2,000 buffer rows per class per accepted update |
| Teacher | T = E0 model, frozen, sha256 checked, as in s3 |
| Seeds | 42, 43, 44 (training randomness, buffer draws and poison draws) |

## Attack (registered budget)

- **Target:** Syn, the E0 attack class (T knows it). **Flip to:** BENIGN. **Where:** stored labels in the replay
  buffer only (D, val, test, T and gradients are out of reach, as before).
- **Per update:** before each update t = 1..6, the attacker flips rho x 2,000 more stored Syn rows (rho = 0.05 ->
  100 rows; rho = 0.10 -> 200 rows). They are drawn uniformly from the Syn rows that are not yet flipped.
- **Flips persist:** a flipped row stays flipped for the rest of the run. Repair changes only the weights / labels
  R1 trains on; it never writes back to the buffer.
- **Cumulative budget:** t x rho of the stored Syn rows after update t: 30% (rho = 0.05) or 60% (rho = 0.10) after E6.
- One attacked class only. No other class is corrupted.

## Arms (per rho and seed, one s3 run, shared T and A0)

- **clean:** no corruption; C1 only.
- **poisoned:** corruption as above; C1 only (gate, fallback).
- **poisoned_repair:** corruption as above; C1, then the **registered soft R1** if C1 is rejected (not relabel; FU2
  results cannot change this).
- **Diagnostic:** forced soft R1 on the poisoned arm at every update (`--forced-r1 soft`): logged only, never a
  decision, never deployed. It shows what repair would do in the cases where the gate did not trigger it.

## Recorded trajectory (validation, every update, every arm)

Primary: the deployed model's Syn val recall after each update t = 0..6. Secondary: Syn mean p(Syn | Syn val rows),
benign FAR, recall of every learned class, gate decision, gate margin and binding constraint, the cumulative share of
flipped stored Syn rows, the C1 candidate's metrics, and the forced-R1 diagnostic's.

Test: only after the validation files are committed, each arm's deployed models A0..A6 are scored once on the
protected test rows (a descriptive test trajectory). No decision uses test.

## Predictions (low confidence)

- **RC1 (ratchet).** At rho = 0.10, the poisoned arm accepts C1 at >= 5 of 6 updates in >= 2 of 3 seeds. Its Syn val
  recall after E6 is lower than A0's by more than 0.05 in >= 2 of 3 seeds, i.e. it drifts past a single update's
  allowance. At rho = 0.05: no prediction about direction.
- **RC2 (silent confidence loss).** Syn mean p(Syn) of the poisoned arm falls with cumulative flips in 3 of 3 seeds at
  rho = 0.10 (from `main`: confidence fell well before recall).
- **RC3 (repair is gated).** Where C1 is never rejected, the repair arm deploys exactly the poisoned arm's models
  (identical weight hashes). Repair cannot act on corruption the gate lets through.
- **RC4 (clean control).** The clean arm accepts all 6 updates and its Syn val recall stays within 0.02 of A0, in
  >= 2 of 3 seeds.

## Stopping rules (fixed before execution)

1. **Feasibility pilot first:** clean arm only, 3 seeds, validation only.
   - Proceed if the clean arm accepts all 6 updates in >= 2 of 3 seeds.
   - Otherwise one registered fallback: drop every class whose clean C1 was rejected in >= 2 seeds, keep name
     order, and rerun the pilot once.
   - Stop and report if it still fails or fewer than 5 updates remain.
2. **Fixed horizon:** every arm runs all 6 updates. No early stop when Syn collapses (that is the result), and no
   extension of the horizon.
3. **Order and budget:** rho = 0.10 first (the stronger case), then rho = 0.05. If the total exceeds 90 min of T4
   time, stop after the finished rho and report.
4. **Invariants:** teacher hash changed, input hash mismatch, or test-row hash mismatch -> stop and report. A
   crashed run resumes with s3's `--resume`.
5. **No changes** to rho, target, schedule, gate or repair method after seeing any poisoned-arm result.

## Analysis and reporting

- Per seed, plot / table of the primary trajectory per arm. Report counts across seeds; no significance claims.
- "Drift" = A0's Syn val recall minus A6's.
- State that the class set was chosen from earlier results, and that RC is CII only and one attacked class.

## Implementation (to be built after this registration; not yet written)

- s3: `--poison-every-update` (default off; the per-update draw above) and `--defer-test` (default off; no test
  scoring inside the run). The smoke test must show that default flags keep every checkpoint hash.
- `s12_repeated.py`: trajectory tables, bundle / restore, and a once-only test command for A0..A6 that checks the
  committed validation files.
- Notebook `08_repeated_corruption.ipynb`.

## Runtime estimate

From sweep / all12 T4 timings, about 7-10 s per attempt:
- pilot: 3 x 6 attempts, about 3 min;
- full runs: 2 rho x 3 seeds x 3 arms x 6 updates, with up to 2 attempts plus 1 diagnostic each, about 30-40 min;
- loading and evaluation: about 10 min;
- total: about 50-60 min of T4 time.

## Deviations

None in the design: rho, target, schedule, arms, gate, repair method, seeds, horizon and stopping rules are unchanged.

### Clarifications before execution (2026-10-27)

These were written together with the code (`s12_repeated.py`, s3 flags, notebook 08), **before any RC run on
real data**. No RC result existed when they were fixed. They pin down details the text above left open.

1. **Attack implementation.** s3 `--poison-every-update` works as follows:
   - **Count.** Before each update t, it flips n = round(rho x the stored Syn rows before E1) more rows. That is
     2,000 stored rows, so n = 100 or 200.
   - **Which rows.** The rows are drawn uniformly from the stored Syn rows that are not yet flipped.
   - **Same rows in both poisoned arms.** Each arm has one draw stream (seed + 2), so `poisoned` and
     `poisoned_repair` flip the same rows. `s12 summarize` asserts this, and that flips persist.
   - **Diagnostic.** The forced soft-R1 diagnostic runs on the poisoned arm only (`--forced-r1 soft
     --forced-r1-arms poisoned`), as registered.
   - **No test.** `--defer-test` makes the runs score no test at all.
2. **Pilot.** The pilot is the clean arm of the rho = 0.10 runs (`rc_rep6_r0.10_s{42,43,44}`, `--arms clean`), and
   the full rho = 0.10 runs resume those runs, so the clean arm is not retrained.
   - The rho = 0.05 runs train their own clean arm, as registered. Its models must be identical to the
     rho = 0.10 clean arm's. That is reported as a determinism check, not as evidence.
   - Fallback tag: `rep6fb`. Stage 1b is recomputed with the remaining classes and the same attack cap.
3. **Deployed model after update t** is the accepted attempt. If both attempts are rejected, it is the previous
   model; its validation numbers are that update's reference evaluation. A0 is t = 0.
4. **Gate margin.** At each attempt, the margins are:
   - old class k: recall_k - (ref_k - 0.05);
   - new class: recall - 0.80;
   - benign FAR: 0.05 - FAR;
   - new-class FAR: 0.05 - new-class FAR.

   The binding constraint is the smallest margin.
5. **RC1.** Both parts must hold in >= 2 of 3 seeds:
   - C1 is accepted at >= (updates - 1) updates (5 of 6);
   - drift (A0 Syn val recall minus A_T's) is > 0.05.
6. **RC2.** "Falls with cumulative flips" means, in a seed, two things:
   - the Spearman correlation over t = 0..T between the deployed model's mean p(Syn | Syn val rows) and the
     cumulative flipped share is < 0;
   - the mean p at A_T is below A0's.

   It is met if both hold in 3/3 seeds at rho = 0.10.
7. **RC3.** In every run (both rho), for every update before the repair arm's first C1 rejection, the repair arm's
   deployed model has the same weight sha256 (state hash) as the poisoned arm's.
8. **RC4.** For >= 2 of 3 seeds, both must hold:
   - the clean arm accepts every update;
   - max over t of |Syn val recall(A_t) - Syn val recall(A0)| <= 0.02.

   It is reported per rho; the two clean arms should be identical.
9. **T4 time** (stopping rule 3) is the summed wall time of the s1b and s3 calls, including the pilot
   (`rc/timing.json`).
   - It is checked after rho = 0.10 and before rho = 0.05 starts.
   - A crashed call is not counted; its rerun is.
10. **Test.** `s12 score-test` runs once, and only when every validation file is byte-identical to the decisions
    commit.
    - It scores each arm's deployed A0..A6 on the protected test rows.
    - It asserts that the test-row hash is the same for every model.
    - The test trajectory is descriptive. No decision uses it.

## Outcome (added 2026-10-28; the design above is unchanged)

Stopped by stopping rule 1 at the clean feasibility pilot (Kaggle, 2026-10-27, code `0d2c31c`):
- 0/3 seeds accepted all six updates;
- the fallback would leave 4 updates (< 5), so the fallback pilot was not run;
- no poisoned or repair arm was trained, and no test was scored.

Files: `results/08_rc/`. RC1-RC4 remain untested. A follow-up with another schedule must be a new registration.
