# Part B: 12-label runs (exploratory, one seed, capped data)

Runs `all12_cii_s42` and `all12_ci_s42` come from notebook 04 part B at code `3e7807c`
(executed notebook: `../notebook04_run2.ipynb`, cell 14). There is no pre-registration with hypotheses: the
schedule rule was committed in `8b4c0b3`, and the gate and thresholds are the same as the sweep's. Test metrics
were computed inside each run, like the sweep's (D1).
Tables are made by `make_result_tables.py` from the saved files:

| File | Content |
|---|---|
| `decisions.csv` | every gate decision (validation): worst old class vs floor, min new-class recall, FARs, reasons |
| `rejected_confusion.csv` | for each rejected attempt, where the failing class's validation rows were predicted |
| `diagnostics.csv` | forced R1 on the clean/poisoned arms: DIAGNOSTIC, never a decision |
| `final_test.csv` | held-out test of the deployed model per arm (+ `fallback_ok` weight-hash check) |
| `schedule.json`, `run_configs.json` | schedule, flags, input sha256 of both runs |

Interpretation: [`../SUMMARY.md`](../SUMMARY.md), section B.
