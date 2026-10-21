# Project memory: poison-resilient continual NIDS (academic capstone)

## Ground rules
- Proposal vs implemented: never describe a planned feature as built or a hypothesis as a result.
  Label statements as: reported by a paper / our inference / our proposal / our code / our experiments.
- No metrics from the old repository or old CIC collection experiments in any new result.
- Dataset: CICDDoS2019 flow files. Schema, class counts and splits come from `nids_out/manifest.json`; do not assume them.
- Leakage rules: split before fitting anything; fit scaler/vocab on E0 TRAIN rows only and freeze (s1b_experiences.py); V and test never used for training,
  replay, or threshold tuning; no SMOTE across splits or tasks. The calib split is created and protected but
  NOT used by any code yet (gate thresholds are fixed constants, not tuned).
- Preserve the class-to-index map (`class_map.json`) across experiences; head expansion copies old rows.
- Attack scope (initial): attacker edits labels of a bounded fraction of ONE old class in the replay buffer B only.
  D, V, test, teacher weights and gradients are out of reach. FGSM/PGD and backdoors are separate threat models.
- Gate: per-old-class recall floors, new-class recall, new-class false-alarm rate. At most two attempts per block (C1, R1).
  Exception, follow-up FU1 only (`s9_followup.py`, `experiments/prereg_followup_fu1.md`): clean blocks get C1, one C2 with a
  validation-selected recipe, and one deferred retry D1; s3 itself is unchanged (a rejected block is skipped for good).
- Everything must run on a Kaggle T4. Smallest meaningful experiment first. Fixed seeds; save configs, manifests, per-round metrics.

- Stage 1 dedupes GLOBALLY on model features (float64 hash) BEFORE capping (bottom-k per class); label-conflict vectors dropped; each feature vector exists once, so no split overlap (asserted; s1b also checks overlap after scaling).
- Benign is scarce (~0.16% raw): keep ALL distinct benign, cap attack rows. CII benign portions are disjoint row ids, never duplicated. Report benign val/test counts with every false-alarm rate.
- Every implementation stage ships BOTH reusable .py code and an explanatory runnable notebook in notebooks/ (notebooks call the scripts, no duplicated logic).
- Inbound is dropped by default (possible label giveaway); ports and ids too. Ablations must be labelled.
- Teacher T = E0 model, saved separately from A0, never trained; sha256 checked at end of every run.
  Repair R1 restarts from the last accepted model (A0 at E1), weights replay rows of T-known classes by p_T(stored label|x).
  Buffer grows only after acceptance; a rejected update keeps the previous model. Head order = arrival order (stored in ckpt).
- Transformed duplicates: s1b excludes calib/val/test rows whose MODEL INPUT (after slog/standardize/clip/float32/vocab,
  -0.0==0.0) equals a training input under the CI or CII scaler; same ids for both scenarios (protected_exclusions.json,
  cause per transform step, per split/class counts, residual must be 0). Never fix this by raising a threshold.
- Comparability: s3 hashes every Stage 1/1b input into config.json and refuses --resume if any changed; final test
  scores BENIGN + all scheduled classes on the same rows for every arm (unlearned class = recall 0; s4 asserts same rows).
- Anything not run on real data is UNVERIFIED. make_synthetic.py output is a plumbing check only.

## Version control
- Git repo, pushed to https://github.com/DevChi1l/cicddos2019-continual-nids (PUBLIC, branch main).
- After every change: commit with a clear message and `git push`. Never commit data, run outputs or checkpoints
  (*.zip, *.parquet, *.npz, *.pt, nids_out/ are gitignored); check for secrets before pushing (repo is public).

## Layout
common.py | s0_audit.py | s1_preprocess.py (clean+split) | s1b_experiences.py (CI/CII + scaler; --tag writes a second schedule under nids_out/<tag>/) | s2_model.py | s3_continual.py (E0, T/A0, arms, gate, poison, repair) | s4_report.py (tables/plots) | s5_explain.py (read-only explanations/diagnostics) | s6_sweep.py (sweep tables: --decisions val-only, then --test) | s7_deliverables.py (report tables/claims from saved files) | demo.py (3 read-only demo interactions) | make_all_labels_schedule.py | make_result_tables.py (small CSV/JSON result tables for git from saved runs) | s9_followup.py (FU1: probes, selection rule, retry policy; reads finished runs only) | make_synthetic.py | smoke_test.py | experiments/prereg_sweep.md, prereg_followup_fu1.md | docs/slides_outline.md | notebooks/00-05
- s3 flags added 2026-10-21, all default-off (default flags reproduce `main`; smoke_test checks checkpoint hashes against the old code):
  --seed (training only), --repair-rule {soft,threeway}, --relabel-to-teacher (a different method), --tw-keep/--tw-quarantine/--tw-mid-weight,
  --forced-r1 (DIAGNOSTIC R1s on clean/poisoned arms -> r1_diag.json, never a decision), --gate-conf-drop (mean p always logged), --tag.
  s3 refuses to write into a completed run (run_manifest.json exists) unless --resume: `main` can never be overwritten.

## Kaggle inputs and outputs per notebook
Kaggle layout (from notebook 03's output on Kaggle): code dataset `dvdhere/nids-code`, which ALSO contains the cleaned splits
(results_01) at `nids/notebooks/results_01/nids_out`; main's results in `dvdhere/nids-results` at `nids_out/`. Notebooks find
folders by searching for a marker file (`s5_explain.find_dir`), never by a hard-coded path.

| Notebook | Needs (Add Input) | Writes |
|---|---|---|
| 00_audit | CICDDoS2019 CSVs; code | `nids_out/manifest.json` (audit) |
| 01_splits_experiences | CICDDoS2019 CSVs (Stage 1 only); code | `clean_{train,calib,val,test}.parquet`, `split_manifest.json`, `class_map.json`, `features.json`, `{ci,cii}/` (scaler, experience ids, manifest), `protected_exclusions.json` = results_01 |
| 02_train_evaluate | code + results_01 (comes with the code dataset); GPU | `runs/cii/<RUN>/` (T.pt, A0.pt, arms, rounds.json, final_test.json, report/), zip `nids_results.zip` |
| 03_understand_our_model | code (with s5_explain.py) + nids-results (main); results_01 optional (comes with the code dataset); CPU | nothing (read-only; asserts every checkpoint unchanged) |
| 04_poison_sweep | new code (git clone with Internet ON, or a code-dataset version with s6_sweep.py) + code dataset for results_01 (found via clean_train.parquet) + nids-results (main, for H0); GPU | `runs/cii/repro_s42`, `runs/cii/sweep_f{f}_s{seed}`, `sweep/` (decisions, diagnostics, hypotheses, test tables); optional part B `all12/` + `runs/{cii,ci}/all12_*_s42`; zip `nids_sweep.zip` (no parquet, nothing from main) |
| 05_followup_fu1 | new code (Internet ON, git clone) + notebook 04's output (clean_*.parquet, all12/, runs/{cii,ci}/all12_*_s42; found via the all12 run marker); GPU | `followup/{cii,ci}/` (separability.csv, probe_E1/E5 + csv, fu1_retry_*_s42/), `followup/decisions.json`; zips `fu1_val.zip`, later `fu1_test.zip` |
Not verified: 00 and 01 run on Kaggle long before this table was written; their input attachment names are not re-checked.
Notebook 04 and part B have run only in local pieces (setup cell, input-hash check, dry run, zip filter), not end to end on Kaggle.
- Architecture name: FT-Transformer-style tabular Transformer (per-feature numeric tokens + [CLS]); the class name `TabTransformer` is historical, NOT the original TabTransformer.
