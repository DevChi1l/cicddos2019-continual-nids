# 11 CG: confidence-aware gate (POST HOC)

**This whole study is POST HOC.** The confidence gate was designed after the main and RF results showed the recall
gate's confidence blind spot. Pre-registration: `experiments/prereg_confgate.md` (commit `65eda77`, pushed before any
CG run). Nothing in the original claims changes.

| Step | Where | Commit |
|---|---|---|
| Pre-registration + code | `experiments/prereg_confgate.md`, `s15_confgate.py` | `65eda77ee7e9a92fded94860e16a14e87b08b589` |
| Validation (Kaggle `dvdhere/nids-cg-val` v1, T4, 1447 s for s15) | `summary.json`, `runs/cii/cg_*` | decisions commit `ef4adec7ec806d1fedee6502a0305dd51e85bfef` |
| Held-out test, scored once (Kaggle `dvdhere/nids-cg-test` v1, CPU) | `test.json` | this commit |

`summary.json`, `test.json` and the `runs/` files are byte-for-byte copies of the Kaggle outputs. `s15 score-test`
refused to run unless `cg/summary.json` was byte-identical to `summary.json` in `ef4adec` and every model's sha256
matched it. `test.json` records `decisions_commit = ef4adec...`. The one change from the original recipe is
`--gate-conf-drop 0.02` (`conf_drop: 0.02` in summary.json).

## Results

Validation decisions come from `summary.json`; test metrics come from `test.json`. "Accepted" counts updates
(experiences) whose final attempt was accepted. Means are over seeds 42, 43, 44.

| Setting | Arm | Updates accepted (per seed) | R1 accepted / ran | Mean test Syn recall (per seed) | Mean test p(Syn) (per seed) |
|---|---|---|---|---|---|
| RF rho 0.10 | clean | 18/18 (6, 6, 6) | 0/0 | 0.9952 (0.9949, 0.9954, 0.9952) | 0.9922 (0.9904, 0.9933, 0.9930) |
| RF rho 0.10 | poisoned | 0/18 (0, 0, 0) | 0/0 | 0.9951 (0.9951, 0.9951, 0.9951) | 0.9928 (0.9928, 0.9925, 0.9929) |
| RF rho 0.10 | poisoned_repair | 17/18 (6, 5, 6) | 17/18 | 0.9941 (0.9938, 0.9951, 0.9934) | 0.9920 (0.9916, 0.9927, 0.9916) |
| RF rho 0.05 | clean | 18/18 (6, 6, 6) | 0/0 | 0.9952 (0.9949, 0.9954, 0.9952) | 0.9922 (0.9904, 0.9933, 0.9930) |
| RF rho 0.05 | poisoned | 3/18 (0, 1, 2) | 0/0 | 0.9951 (0.9951, 0.9951, 0.9951) | 0.9765 (0.9928, 0.9765, 0.9602) |
| RF rho 0.05 | poisoned_repair | 18/18 (6, 6, 6) | 15/15 | 0.9898 (0.9951, 0.9950, 0.9795) | 0.9883 (0.9937, 0.9939, 0.9772) |
| MAIN f 0.5 | clean | 6/6 (2, 2, 2) | 0/0 | 0.9954 (0.9954, 0.9954, 0.9954) | 0.9945 (0.9945, 0.9945, 0.9946) |
| MAIN f 0.5 | poisoned | 0/6 (0, 0, 0) | 0/0 | 0.9953 (0.9951, 0.9954, 0.9954) | 0.9951 (0.9949, 0.9952, 0.9953) |
| MAIN f 0.5 | poisoned_repair | 5/6 (2, 1, 2) | 5/6 | 0.9946 (0.9951, 0.9951, 0.9936) | 0.9938 (0.9944, 0.9939, 0.9932) |

The "poisoned" (gate-only) arm keeps high Syn confidence because it rejects almost every update and so stays on an
earlier model. In MAIN it therefore never learns the new classes: test recall is LDAP 0.0 and UDP 0.0 in all 3 seeds.

## Hypotheses (POST HOC)

- **CG0 (descriptive): same decisions as the committed clean runs. Observed.** All six CG-RF clean arms and the
  CG-MAIN seed-42 clean arm have the same accept/reject sequence (experience, attempt, accepted) as the committed
  `10_rf/runs/cii/rf_ref6_*/clean/rounds.json` and `02_main/clean/rounds.json`. Every update was accepted.
- **CG1 (no false alarms of the gate): SUPPORTED.** Every clean-arm update was accepted: 36/36 CG-RF (18 at
  rho 0.10, 18 at rho 0.05) and 6/6 CG-MAIN. No clean update was rejected, so CG claims are allowed in both settings.
- **CG2 (the gate now sees the poison): SUPPORTED.** The poisoned C1 at the first update (E1) was rejected in 3/3
  seeds at rho 0.10 (val mean p(Syn) 0.9509, 0.9498, 0.9563 against floors 0.973, 0.972, 0.973) and in 3/3 CG-MAIN
  seeds (0.7895, 0.7663, 0.8438 against floors 0.975, 0.976, 0.976). This hypothesis did not cover rho 0.05. There
  the E1 poisoned C1 was rejected only in seed 42 and **accepted** in seeds 43 and 44 (val mean p(Syn) 0.9767 and
  0.9737, both above the 0.02 floor). The gate's rejections began at E2 (s43) and E3 (s44).
- **CG3 (repair restores confidence): SUPPORTED.** The deployed poisoned_repair model's mean test p(Syn) was >= 0.95
  in all 9 runs: RF rho 0.10: 0.9916, 0.9927, 0.9916; RF rho 0.05: 0.9937, 0.9939, 0.9772; MAIN: 0.9944, 0.9939,
  0.9932. The lowest was 0.9772 (RF rho 0.05 s44, test Syn recall 0.9795).
- **CG4 (still learning): NOT SUPPORTED (it holds for CG-RF and fails for CG-MAIN).**
  - CG-RF: poisoned_repair accepted >= 5 of 6 updates in every run (6, 5, 6 at rho 0.10; 6, 6, 6 at rho 0.05). That
    part holds. In rho 0.10 s43, the E3 R1 repair was rejected (val mean p(Syn) 0.9647 < floor 0.975).
  - CG-MAIN: all four classes were learned in only **2/3** seeds. In seed 43, the E1 R1 repair was rejected (val
    mean p(Syn) 0.9680 < floor 0.976), so LDAP was never added. The deployed model's classes are BENIGN, Syn, UDP,
    and its test LDAP recall is 0.0. Seeds 42 and 44 learned all four classes (test LDAP 0.9880 / 0.9884,
    UDP 0.9953 / 0.9975).

## Scope

POST HOC, one attacked class (Syn → BENIGN), 3 seeds, no significance claim. One execution per setting; no
threshold search and no re-runs. Neither kernel crashed and nothing was resumed. An adversary who keeps confidence
high was not tested.
