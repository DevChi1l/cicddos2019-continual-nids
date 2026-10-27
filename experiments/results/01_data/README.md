# Stage 1 / Stage 1b provenance copies (no data rows)

These are **copies** of the small JSON files that notebook 01 (Stage 1 + Stage 1b) wrote on Kaggle. They contain no
flow rows and no model. They are committed so that every count in `docs/PROJECT_TECHNICAL_HANDBOOK.md` and
`docs/RESULT_TABLES.md` can be traced to a file in git.

Source: `notebooks/results_01.zip` (sha256 `0a11a51c…`, not committed). Every file below is byte-identical to the
copy inside `nids_results.zip` (the `main` run), and the five that s3 hashes match the input hashes recorded in
`experiments/prereg_sweep.md` and in `02_main/run_manifest.json`.

| File | What it is |
|---|---|
| `split_manifest.json` | Stage 1: arguments, raw / non-finite / distinct / conflict / kept counts per label, rows per file, split counts (sha256 `bb1f5acb…`) |
| `class_map.json` | the 14 raw labels -> indices (data side) |
| `features.json` | the 77 numeric model features and the 1 categorical (`protocol`) |
| `schedule.json` | the four-label schedule: E0 Syn, E1 LDAP, E2 UDP |
| `protected_exclusions.json` | Stage 1b: the 166 calib/val/test row ids whose scaled model input equals a training input, with causes and the zero-residual check |
| `cii/experience_manifest.json`, `ci/experience_manifest.json` | Stage 1b: rows per experience, attack cap 13,890, protected rows kept per split and class |
| `cii/scaler.json`, `ci/scaler.json` | the frozen E0-only scaler (mean, std, clip 5, protocol vocabulary) |

The raw-data audit (notebook 00) is tracked separately as `notebooks/manifest.json` (18 files, 70,427,637 rows).
