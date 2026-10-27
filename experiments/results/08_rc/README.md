# RC (repeated small replay corruption): STOPPED at the registered feasibility pilot

Pre-registration: `experiments/prereg_repeated_corruption.md`. Code: commit `0d2c31c` (`s12_repeated.py`, notebook 08).
Kaggle T4 session, 2026-10-27 (18:12-18:17 UTC); notebook 08 cells 1-3 only.

**Outcome: `PROCEED = False`. No poisoned or repair arm was trained, no RC prediction (RC1-RC4) was tested, and no
test row was scored.** Cells 4-9 were not run. This experiment is closed. Any follow-up with a different schedule is a separate
registration; nothing here may be presented as its result, or the reverse.

## Files (unchanged from `results_8.zip`, sha256 bc9c2fc8…; byte-checked)

| File | What it is |
|---|---|
| `rc/pilot.json` | the registered pilot decision (written by `s12_repeated.py pilot`) |
| `rc/timing.json` | measured T4 time of the s1b and s3 calls: 202 s in total |
| `rc/schedule_rep6.json` | the registered schedule: E0 Syn, then LDAP, MSSQL, NTP, SNMP, SSDP, TFTP |
| `rep6/cii/experience_manifest.json`, `rep6/protected_exclusions.json` | Stage 1b of tag `rep6`; byte-identical to the local feasibility check made before registration |
| `runs/cii/rc_rep6_r0.10_s{42,43,44}/` | `config.json`, `run_manifest.json`, `A0_metrics.json`, `clean/rounds.json`, `clean/val_done.json` of the three clean pilot runs (no checkpoints, no buffers) |

Checks:
- the protected val rows equal the registered feasibility check (printed by `prepare`);
- the teacher is unchanged in all 3 runs (`T_unchanged: true`);
- every run has `test_scored: false`, and there is no `A0_test`;
- the code inside the zip equals commit `0d2c31c`.

## Pilot result (validation, clean arm only)

| Update | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| E1 LDAP | accepted | accepted | accepted |
| E2 MSSQL | accepted (LDAP margin +0.037) | **rejected**: LDAP 0.929 < floor 0.940 | **rejected**: LDAP 0.906 < floor 0.935 |
| E3 NTP | accepted | accepted | accepted |
| E4 SNMP | **rejected**: LDAP 0.556 < floor 0.925 | **rejected**: LDAP 0.525 < floor 0.935 | **rejected**: LDAP 0.524 < floor 0.936 |
| E5 SSDP | accepted (LDAP margin +0.000) | accepted | accepted |
| E6 TFTP | accepted | accepted | accepted |

Read from `clean/rounds.json`. A margin is the candidate's recall minus the gate floor (last accepted model's recall − 0.05).

- **Rule applied as registered.**
  - 0 of 3 seeds accepted all six updates, and ≥ 2 were needed.
  - The fallback drops every class rejected in ≥ 2 seeds, which is MSSQL (2 seeds) and SNMP (3 seeds).
  - That leaves 4 updates (LDAP, NTP, SSDP, TFTP), fewer than the minimum of 5, so the rule said stop. The fallback
    pilot was therefore not run.
- **Every rejection was caused by old-class LDAP recall.** Where the rejected candidates sent the LDAP val rows:
  - SNMP update: 39-47% of LDAP rows went to SNMP;
  - MSSQL update: 6-8% went to MSSQL.

  At E4, SNMP's new-class FAR was also close to its limit (0.038-0.046 vs 0.05), mostly LDAP rows predicted as SNMP.
- **One acceptance was knife-edge.** Seed 42 accepted E5 with an LDAP margin of +0.000 (0.926 vs floor 0.925).
- **Syn (the class RC would attack) did not move in the clean arm.** Its val recall stayed between 0.995 and 0.996 at
  every update in every seed. This is descriptive only; RC4 was not evaluated, because no full run exists.

### What this does and does not say

The tested recipe cannot learn this class sequence under the gate in this order: once LDAP is learned, SNMP (and,
less strongly, MSSQL) erases it. It says nothing about replay poisoning; no poison was applied.
