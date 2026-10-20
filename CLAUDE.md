# Project memory: poison-resilient continual NIDS (academic capstone)

## Ground rules
- Proposal vs implemented: never describe a planned feature as built or a hypothesis as a result.
  Label statements as: reported by a paper / our inference / our proposal / our code / our experiments.
- No metrics from the old repository or old CIC collection experiments in any new result.
- Dataset: CICDDoS2019 flow files. Schema, class counts and splits come from `nids_out/manifest.json`; do not assume them.
- Leakage rules: split before fitting anything; fit scaler/vocab on E0 TRAIN rows only and freeze (s1b_experiences.py); V and test never used for training,
  replay, or threshold tuning (thresholds use the calibration split); no SMOTE across splits or tasks.
- Preserve the class-to-index map (`class_map.json`) across experiences; head expansion copies old rows.
- Attack scope (initial): attacker edits labels of a bounded fraction of ONE old class in the replay buffer B only.
  D, V, test, teacher weights and gradients are out of reach. FGSM/PGD and backdoors are separate threat models.
- Gate: per-old-class recall floors, new-class recall, new-class false-alarm rate. At most two attempts per block (C1, R1).
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

## Layout
common.py | s0_audit.py | s1_preprocess.py (clean+split) | s1b_experiences.py (CI/CII + scaler) | s2_model.py | s3_continual.py (E0, T/A0, arms, gate, poison, repair) | s4_report.py (tables/plots) | s5_explain.py (read-only explanations/diagnostics) | make_synthetic.py | smoke_test.py | notebooks/00_audit, 01_splits_experiences, 02_train_evaluate, 03_understand_our_model (read-only)
- Architecture name: FT-Transformer-style tabular Transformer (per-feature numeric tokens + [CLS]); the class name `TabTransformer` is historical, NOT the original TabTransformer.
