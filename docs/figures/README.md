# Figure index (generated from committed files)

Regenerate:
- `python make_figures.py` writes every figure;
- `python make_figures.py --check` verifies them;
- `python make_figures.py --execute-notebook` runs `notebooks/11_results_storyboard.ipynb` end to end;
- On Kaggle: import `notebooks/11_results_storyboard.ipynb`, set Internet **On** and Accelerator **None (CPU)** with no
  datasets, then *Run All*. The first cell clones this public repository into `/kaggle/working/` (rerun-safe) and
  reads committed files only.

Every PNG has a `*_data.csv` with exactly the plotted numbers (the table view). `--check` rebuilds the figures and
asserts that those numbers equal the committed result tables.

**Conventions:**
- Each panel names its evidence: **VALIDATION**, **HELD-OUT TEST** or **DATA PROVENANCE**.
- Seeds are 42-44 unless the panel says one seed.
- Bars and whiskers show the mean and the min-max over seeds; hollow dots in figure 2 are individual seeds.
- Only measured points are drawn. In figure 6, lines only join measured update points; nothing is interpolated or
  smoothed.
- The y-axes in figure 6 are truncated and labelled so.

| Figure | Evidence | Claim it supports | Limitation it shows or must carry | Sources |
|---|---|---|---|---|
| `fig1_data_pipeline.png` | data provenance | All 70,427,637 raw rows (18 files, 14 labels) were read once. Global dedupe and label-conflict removal shrink each class. BENIGN is 0.16% and never capped | Training used capped subsets (98,833 rows four-label; 159,595 max). Filenames are not labels | `notebooks/manifest.json`, `experiments/results/01_data/split_manifest.json` |
| `fig2_sweep.png` | validation (top left), held-out test (others) | **Pre-registered** sweep: rejections rise with f. Soft repair keeps all four classes learnable up to f = 0.95 | At f = 1.0 soft repair fails (0/6). The gate alone keeps a stale model (LDAP/UDP = 0). Gate-only results are bimodal across seeds | `04_sweep/decisions.csv`, `04_sweep/final_test.csv` |
| `fig3_fu2_defence.png` | held-out test | **Primary result:** at f = 1.0 teacher relabelling learns all four classes in 3/3 seeds, within about 0.01 of clean, at benign FAR about 0.0011 | **Post hoc** method; E1 replays the sweep diagnostic. Four labels, one attacked class, 3 seeds. A FAR of 0 for gate-only is not safety | `06_fu2/test/test_recall.csv` |
| `fig4_12label_vs_oracle.png` | held-out test | The CII continual 12-label models retain 6 of 12 classes. A jointly trained oracle reaches all 12 (8 at >= 0.80) | The oracle is a diagnostic, **not** a continual or defended result. The 12-label continual goal is unsolved. Continual runs are single-seed | `04b_all12/final_test.csv`, `05_fu1/test/test_recall.csv`, `07_oracle/test/test_recall.csv` |
| `fig5_rc_rcs_pilots.png` | validation only | Clean RC/RCS rejections are old non-Syn classes forgotten when an overlapping class arrives. Syn stays at 0.993-0.996 | Both experiments **stopped** at their registered clean stage. No poisoned run and **no test** exist. Only three schedules were tried | `08_rc/runs/*/clean/rounds.json`, `09_rcs/runs/*/clean/rounds.json`, `pilot.json`, `screen.json` |
| `fig6_rf_trajectory.png` | validation (top), held-out test (bottom) | At 10% more flips per update (60% after six updates), Syn recall holds at about 0.994 while mean Syn confidence falls to about 0.77. No update was rejected and repair never triggered | **Post hoc**; three fixed classes, traffic refresh, not new-class arrival. The confidence loss was **not** repaired | `10_rf/rf/trajectory_val.csv`, `10_rf/test/rf/test/test_trajectory.csv` |
| `fig7_status.png` | status | Which experiments were pre-registered, post hoc, validated, tested once or stopped. Each cell is asserted against the evidence folders in git | `*`: for `main`, the sweep and all12, test was computed inside the run before a decisions commit (disclosed, never used) | `experiments/results/*`, `experiments/prereg_*.md` |
