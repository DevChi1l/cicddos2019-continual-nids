# Sweep results, part 1: validation decisions (notebook 04, Kaggle)

Pre-registration: [`../../prereg_sweep.md`](../../prereg_sweep.md) (commit `eb6729f`). This folder holds the
**validation-only** outputs, committed before any test table is made. Test tables come in a later commit.

## Files (exact bytes from the Kaggle run; sha256 in `SHA256SUMS`)

| File | What it is |
|---|---|
| `decisions.csv` | `s6_sweep.py --decisions`: one row per (run, arm, experience, attempt), val metrics only |
| `diagnostics.csv` | forced R1 (`r1_diag.json`): DIAGNOSTIC, never a decision, never deployed, no test score |
| `hypotheses.csv` | H0-H6 evaluated from the two files above |
| `decisions.png` | poisoned C1 at E1: Syn val recall per seed vs its gate floor |
| `cell9_output.txt` | notebook cell 9 stdout (the printed val tables) |
| `training-04.ipynb` | the executed notebook (Kaggle "Save & Run All", papermill timestamps in its metadata) |

Source: `notebooks/results_04.zip` (Kaggle output of that version, sha256 `fd2fdf98dfe7...`), which contains
`nids_out/` and `nids_sweep.zip` (sha256 `223d1fad76b5...`).

## Checks done before this commit (2026-10-21)

- Code: the clone used by the run is commit `8b4c0b3` (printed by cell 1).
- Inputs: all 13 runs (`repro_s42` + 12 `sweep_*`) hash the same 9 Stage 1/1b inputs as the prereg table;
  the flags match the prereg grid (poison_frac, seed, `--forced-r1 soft,threeway,threeway+relabel`,
  repair rule soft, thresholds 0.9 / 0.1 / 0.5, `--gate-conf-drop` unset); teacher T unchanged in every run.
- The three CSVs in `nids_out/sweep/` and inside `nids_sweep.zip` are identical (sha256).
- Recomputed `s6_sweep.py --decisions` from the saved `config.json`/`rounds.json`/`r1_diag.json` into a
  scratch folder (not the results): all three CSVs byte-identical, printed tables identical to cell 9.

## Protocol deviation D1 (recorded 2026-10-21): test scores were computed before this commit

Code path (verified in `s3_continual.py` at `8b4c0b3`):
`run_arm()` ends with `evaluate(A, ..., ctx["test"])` on the deployed model and writes
`runs/<scenario>/<run>/<arm>/final_test.json`; `main()` writes `A0_test` into `run_manifest.json`.
In notebook cell 7, `s4_report.py` then wrote `report/final_test.csv` for each run.
`s6_sweep.py --test` does not score anything: it only collects those existing `final_test.json` files.

| Step | What happened | When (UTC, from papermill metadata / git) |
|---|---|---|
| Pre-registration committed | `eb6729f`; tooling `8b4c0b3` | 2026-10-20 22:32 / 22:38 |
| **Computation** of test metrics | inside cell 7 (s3 `final_test.json`, `run_manifest.json` `A0_test`; s4 `report/final_test.csv`) for all 13 runs | 2026-10-20 23:12-23:55 |
| Test files packaged | cell 15 zipped them into `nids_sweep.zip` (52 test files); Kaggle saved the whole output | 2026-10-20 23:55 |
| **Display** in the notebook | none: cells 5/7 print val-only tables (s3 log lines, s4 attempts table), cell 9 prints val only, cell 11 (`s6 --test`) was skipped | - |
| **Human inspection** | the user reports looking only at the val tables. Whether a file was opened cannot be verified from the files. Claude did not read any test value before this commit | - |
| **Reporting** (test tables `final_test.csv`, `test_summary.csv`) | not yet; only after this commit | after this commit |

So the prereg's rule 2 ("committed before `s6 --test`") holds only in the literal sense: no test **table**
existed and no test number was shown, but test **scores** were computed and saved about 1.5 h before this
commit. We do not claim that the first test scoring happened after the commit. Nothing was deleted or
regenerated. The decisions themselves use val only (`rounds.json` is written before the test evaluation, and
nothing in the gate reads test rows), so this deviation does not affect H0-H6.
Wording affected: the STOP cell in notebook 04 ("before any test number is produced") is inaccurate.
