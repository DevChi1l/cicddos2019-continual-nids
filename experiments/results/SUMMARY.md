# Results summary: poison sweep, 12-label runs (notebook 04) and follow-up FU1 (notebook 05)

All numbers below come from saved run files. The tables in this folder are produced by
`python make_result_tables.py --out <nids_out>`, which reads files and trains nothing. Each section says
whether a number is a **validation gate decision** (used by the method) or a **held-out test metric**
(scored once on the deployed model, never used by any decision).

Provenance: notebook 04 at code commit `3e7807c` (Kaggle T4, 2026-10-21 10:35-11:37 UTC), executed notebook
`notebook04_run2.ipynb`. Its 13 four-class runs are bit-identical to the first execution, whose validation
files were committed in `eebc990` before any test table was made. Every checkpoint, metric and table is
identical; only wall-clock `seconds` differ. Deviations D1 and D2 are in `../prereg_sweep.md`.

## Setup in one paragraph

A tabular Transformer learns in experiences: E0 = BENIGN + Syn, then new attack classes arrive. It replays a
buffer of 2,000 stored rows per class. The attacker flips the stored label of a share f of the Syn buffer rows
to BENIGN, once, before E1, and the flips persist. Every update (C1) must pass a **validation gate**:
- old-class recall must stay within 0.05 of the last accepted model;
- new-class recall must be at least 0.80;
- benign false-alarm rate (FAR) and new-class FAR must be at most 0.05.

If C1 fails, the `poisoned_repair` arm tries one repair, R1. R1 retrains with each buffer row weighted by
p_T(stored label), where T is the frozen E0 model. If R1 also fails, the previous model stays deployed. The
three arms are:
- `clean`: no poison;
- `poisoned`: poison, no repair;
- `poisoned_repair`: poison plus R1.

## A. Four-class sweep (CII, BENIGN/Syn -> LDAP -> UDP), f in {0.75, 0.9, 0.95, 1.0}, seeds 42-44

Pre-registered (`../prereg_sweep.md`). Three seeds per f, so we report counts and ranges only, with no
significance tests.

**Validation gate decisions** (`04_sweep/decisions.csv`, `hypotheses.csv`):

| f | poisoned C1 at E1 rejected (seeds) | R1 ran / passed (`poisoned_repair`, E1+E2) |
|---|---|---|
| 0.75 | 1/3 | 1 / 1 |
| 0.9 | 2/3 | 4 / 4 |
| 0.95 | 3/3 | 6 / 6 |
| 1.0 | 3/3 | 6 / **0** |

- The clean arm's C1 was accepted at all 24 updates.
- `repro_s42` made the same decisions as `main`.
- H1 ("f <= 0.9 never rejected") is **not supported**: rejection already starts at f = 0.75 in some seeds.

**Held-out test, deployed model after E2** (`04_sweep/deployed_models.csv`, `final_test.csv`, `test_summary.csv`):

| Arm | Runs | Learned all 4 classes | Syn | LDAP | UDP | Benign FAR |
|---|---|---|---|---|---|---|
| clean | 12 | 12 | 0.9954 | 0.983-0.989 | 0.998-0.999 | 0.0005-0.0018 |
| poisoned, f <= 0.95 | 9 | **3** | 0.960-0.995 | 0 or 0.983-0.988 | 0 or 0.999-1.000 | 0-0.0013 |
| poisoned_repair, f <= 0.95 | 9 | **9** | 0.980-0.995 | 0.975-0.988 | 0.997-1.000 | 0.0007-0.0018 |
| poisoned and poisoned_repair, f = 1.0 | 3 + 3 | **0** (A0 kept) | 0.995 | 0 | 0 | 0 |

- The fallback works as specified: the weight hashes confirm that every rejected update left the previous
  model deployed.
- Without repair, the gate protects Syn but the model **stops learning**. LDAP is never learned in 9 of 12
  poisoned runs and UDP in 8 of 12.
- A Syn-only test plot (`test_syn.png`) hides this, because a frozen A0 still detects Syn.
- A benign FAR of 0 there is not a safety result: A0 predicts BENIGN for LDAP and UDP traffic.

### Why repair works for f <= 0.95

- **T separates the flipped rows almost perfectly.** It gives them mean weight 0.003, against 0.997-0.999 for
  clean rows (H4: 24/24). So R1 effectively deletes the wrong labels.
- **Some correct Syn labels survive.** There are still 500, 200 or 100 correctly labelled Syn rows
  (f = 0.75 / 0.9 / 0.95), and the class-balanced loss up-weights that small class.
- **Result:** Syn validation recall comes back above its floor in all 11 R1 attempts (0.964-0.996), and the
  final models learn all four classes.
- **Limit:** R1 reweights; it does not clean the buffer. The flips stay, so in 5 runs R1 had to run again at E2.

### Why repair fails at f = 1.0

- Every stored Syn row is labelled BENIGN, and E1/E2 contain no Syn data.
- Soft R1 down-weights all 2,000 flipped rows to about 0.003. That removes the wrong labels, but it leaves
  essentially **no Syn label at all**.
- So the R1 model forgets Syn: validation Syn recall 0.000-0.0002 in all 6 attempts, and the gate rejects it
  (seed 44, E1 also failed benign FAR 0.058 and new-class FAR 0.42).
- The fallback then keeps A0 at every update, so LDAP and UDP are never learned.
- In short, weighting can remove wrong labels but cannot supply correct ones.

*Diagnostic only (not the decision rule, never deployed, no test score):*
- At f = 1.0, a forced "three-way + relabel" repair would have passed the gate 6 of 6 times. It replaces the
  stored label with T's prediction, and 1,994-1,995 rows were relabelled, all to their true class.
- Its 0.9 / 0.1 thresholds were chosen after seeing `main`.
- This is a candidate method, not a result.

## B. 12-label runs `all12_cii_s42` and `all12_ci_s42` (exploratory: one seed, capped data)

**Design and comparability:**
- **Single seed (42).** No hypotheses were pre-registered; the schedule rule was committed in `8b4c0b3`
  before the run.
- **Capped data.** Stage 1 kept at most 60,000 distinct rows per class, then 9,312 rows per attack class.
- **Two labels excluded.** Portmap (363 train / 60 val) and WebDDoS (132 / 22) have too little support.
- **Schedule.** E0 = BENIGN + Syn, then two classes per update in name order (`04b_all12/schedule.json`).
- **Benign split.** CII gives each update a disjoint benign portion of about 9,527 rows; CI puts all 57,163
  benign rows in E0.
- **Shared test rows.** CI and CII share every attack row and the same protected test rows: 123,281 rows, 444
  excluded as transformed duplicates.
- **Not comparable to `main` or the sweep.** It refits the scaler and uses different rows, so these numbers
  are not row-comparable with `main` or the sweep.
- **Poison:** f = 0.5 on Syn.

**Validation gate decisions** (`04b_all12/decisions.csv`, `rejected_confusion.csv`). Clean updates were rejected
too, so these rejections are not caused by the poison:

| Update | CII clean | CI clean | Poisoned arms (both scenarios) | Reason (validation) |
|---|---|---|---|---|
| E1 (DNS, LDAP) | rejected | rejected | rejected | new DNS recall 0.58-0.73 < 0.80; 26-41% of DNS rows predicted LDAP |
| E2 (MSSQL, NTP) | accepted | accepted | accepted | |
| E3 (NetBIOS, SNMP) | rejected (MSSQL 0.9390 vs floor 0.9396) | accepted (MSSQL 0.93306 vs floor 0.93300) | rejected | old MSSQL recall below floor (rows to SNMP/NTP/NetBIOS) |
| E4 (SSDP, TFTP) | accepted | accepted | accepted | |
| E5 (UDP, UDPLag) | rejected | rejected | rejected | old SSDP recall <= 0.009: 98-99% of SSDP rows predicted UDP; new-class FAR 0.11-0.14 |

- **Soft R1 rescued none of the 6 rejected poisoned updates** (E1, E3, E5 in both scenarios).
- **R1 can't reach these failures by design.** T is the E0 model and knows only BENIGN and Syn, so R1 only
  reweights BENIGN/Syn buffer rows. Every failure here concerns other classes: DNS vs LDAP, MSSQL, SSDP vs UDP.
- **Syn was never the problem.** Its validation recall stayed at or above 0.944 in every attempt.
- At E3 the clean arms sit within 0.001 of the MSSQL floor in both scenarios, so with one seed the clean CI
  accept vs CII reject is not interpretable. The poisoned arms missed it by more (CII 0.016, CI 0.090), but
  one seed cannot attribute that to the poison.

**Held-out test, deployed model after E5** (`04b_all12/final_test.csv`):

| | CII clean | CII poisoned / repair | CI clean | CI poisoned / repair |
|---|---|---|---|---|
| Classes with recall > 0 | BENIGN, Syn, MSSQL, NTP, SSDP, TFTP | same | + NetBIOS, SNMP | BENIGN, Syn, MSSQL, NTP, SSDP, TFTP |
| Recall of learned attack classes | 0.956-0.996 | 0.957-0.996 | 0.937-0.994 | 0.969-0.996 |
| DNS, LDAP, UDP, UDPLag (+ NetBIOS, SNMP where E3 was rejected) | 0 | 0 | 0 | 0 |
| Benign FAR | 0.0010 | 0.0013 | **0.0290** | 0.0048 |
| Macro recall (12 classes) | 0.492 | 0.492 | 0.648 | 0.492 |

- The poisoned and repair arms deployed the same model in each scenario (R1 never passed), so their test rows
  are identical.

**Reading:** with these fixed gate thresholds, the update protocol does not scale to 12 labels. Some class
pairs (DNS/LDAP, SSDP/UDP) are hard to separate with these features. A rejected update is never retried, so
its classes are lost permanently. This is a property of the protocol, observed in one seed; it is not a
poisoning result. Thresholds were **not** changed after seeing these test numbers.

## C. Follow-up FU1: retrying the rejected clean blocks of part B (pre-registered, one start seed)

Details, provenance and tables: [`05_fu1/README.md`](05_fu1/README.md). Pre-registration:
`../prereg_followup_fu1.md`. Decisions on validation (committed `5307b58`), test scored once afterwards (2026-10-27).

- **Validation:** 20 update epochs (`ep20`) let E1 (DNS, LDAP) pass in 5 of 6 probe cells, so the retry policy
  used it. No recipe passed E5 (SSDP vs UDP). In the full schedule only E1 (at C2) and E4 were accepted, in both
  scenarios.
- **Test, deployed model (123,281 rows, same rows as part B):**

  | | CII s3 clean | CII FU1 | CI s3 clean | CI FU1 |
  |---|---|---|---|---|
  | Labels learned (of 12) | 6 | 6 | 8 | 6 |
  | DNS / LDAP recall | 0 / 0 | 0.831 / 0.834 | 0 / 0 | 0.837 / 0.829 |
  | Lost vs s3 (recall -> 0) | | MSSQL, NTP | | MSSQL, NTP, NetBIOS, SNMP |
  | Macro recall (12 labels) | 0.492 | 0.471 | 0.648 | 0.470 |
  | Benign FAR | 0.0010 | 0.0018 | 0.0290 | 0.0084 |

- **Reading:** FU1 makes DNS/LDAP learnable, but once they are accepted their recall floors block the next
  blocks, which overlap with them. Label coverage does not grow, and macro recall falls. SSDP/UDP stays unsolved.
  This is a negative result for "better update recipes fix the 12-label protocol" (one start seed).

## What we can and cannot claim

| Claim | Status |
|---|---|
| Teacher-weighted soft repair restores Syn and lets the model learn LDAP/UDP when f <= 0.95 | validation decisions + test, 3 seeds per f, CII only |
| Soft repair fails when no correct label of the attacked class remains (f = 1.0) | 6/6 attempts, mechanism explained above |
| The gate stops corrupted updates, but without repair the model stops learning | 9/12 runs miss LDAP |
| Relabel-to-teacher fixes f = 1.0 | **diagnostic only**, not deployed or tested |
| The method works with 12 labels | **no**: clean updates fail the gate (one seed, capped data) |
| Retrying rejected clean blocks with a validation-selected recipe (FU1) fixes it | **no**: DNS/LDAP learned (test 0.83), but 6 of 12 labels in both scenarios, macro recall lower |
| Mild (5-10%) repeated corruption | not run |
| Statistical significance | not claimed (3 seeds, or 1 for part B) |
