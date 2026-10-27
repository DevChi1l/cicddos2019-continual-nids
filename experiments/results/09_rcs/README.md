# RCS (repeated corruption on a stability-screened schedule): STOPPED at Stage S

Pre-registration: `experiments/prereg_rcs.md` (commit `5e4ada3`). Code: commit `c33e961` (`s13_rcs.py`, notebook 09).
Kaggle T4, notebook 09 cells 1-3 only.

**Outcome: `SELECTED = None`.** Neither candidate was stable in any clean screening seed, so stopping rule 1 applied:
- no confirmation run (seeds 45-47);
- no poisoned or repair arm;
- no test.

RCS is closed. It is separate from RC (`08_rc/`), which also stopped at its clean pilot.

## Files (unchanged from `rcs_clean.zip`, sha256 368f870d…; byte-checked)

| File | What it is |
|---|---|
| `rcs/screen.json` | the registered Stage S decision (written by `s13_rcs.py screen`) |
| `rcs/timing.json` | measured T4 time: 406 s in total (2 Stage 1b tags plus 6 clean runs of 56-70 s each) |
| `rcs/schedule_rcs{6,5}.json`, `rcs{6,5}/cii/experience_manifest.json` | the two registered candidates |
| `runs/cii/rcs_rcs{6,5}_r0.10_s{42,43,44}/` | `config.json`, `run_manifest.json`, `A0_metrics.json`, `clean/rounds.json`, `clean/val_done.json` |

Checks:
- both Stage 1b manifests are byte-identical to the feasibility check made before registration;
- the candidates' classes equal `s13_rcs.CANDIDATES`;
- recomputing the stability predicate from `rounds.json` reproduces `screen.json` exactly;
- the teacher is unchanged in every run, `test_scored` is false everywhere, and there is no `A0_test`.

## Screen result (clean arm, validation)

The table shows the rejected updates. Every other update was accepted.

| Candidate | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| `rcs6` (… TFTP, LDAP last) | E3 SNMP: old MSSQL 0.867 < 0.926; E6 LDAP: old MSSQL 0.906 < 0.927 | E6 LDAP: old SNMP 0.654 < 0.928 | E6 LDAP: old SNMP 0.451 < 0.926, new-class FAR 0.069 |
| `rcs5` (LDAP dropped) | E3 SNMP: old MSSQL 0.892 < 0.928 | E3 SNMP: old MSSQL 0.923 < 0.932; E4 SSDP: old NTP 0.904 < 0.944 | E3 SNMP: old MSSQL 0.900 < 0.924 |

- **Stable seeds: 0/3 for both candidates.** Condition (a), every update accepted, already failed in every run.
- **Syn**, the class the attack targets, stayed at 0.995-0.996 in all 6 runs (max change 0.0007).
- **Moving LDAP last moved the failure, it did not remove it.** Learning LDAP erased SNMP: 2/3 seeds, recall 0.45-0.65.
- **Dropping LDAP exposed the next overlap.** SNMP erased MSSQL in 3/3 seeds, and SSDP erased NTP once.

## Reading RC and RCS together (9 clean runs, validation only)

Corrected 2026-10-28: an earlier version of this heading said 15. The count from the committed `rounds.json` files is
9 clean runs (RC 3, RCS 6), with 13 rejected updates.

- **Every clean rejection was the same kind of failure.** An old attack class that is *not* Syn is forgotten when
  another amplification class that overlaps it arrives: LDAP/SNMP, MSSQL/SNMP, LDAP/MSSQL, NTP/SSDP.
- **Syn never moved by more than 0.001.**
- **The instability comes from class arrival, not from the attacked class.** Under this gate and recipe, none of
  the three registered schedules (RC's, `rcs6`, `rcs5`) was stable. Other orders and class sets were not tried, so
  this is not a claim about every possible order (wording corrected 2026-10-28).
- **What this does not show.** It says nothing about replay poisoning, since no poison was applied. It is not a claim
  that the classes are inseparable: a jointly trained model learns MSSQL, NTP and SNMP at 0.83-0.99 (`07_oracle/`).
