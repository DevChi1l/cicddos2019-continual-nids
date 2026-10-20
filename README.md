# Poison-resilient continual NIDS - starter pipeline (Stages 0-2)

Status: **Stages 0-4 ran end to end on synthetic CICDDoS2019-style files only (`smoke_test.py`, CPU PyTorch). On real data (Kaggle): the Stage 0 full audit and Stage 1 have run (95,272 distinct benign kept, zero cross-split feature overlap); Stage 1b passed for CI and CII (166 protected rows excluded as transformed duplicates, residual overlap 0; outputs in `results_01.zip`); one CII run (seed 42, poison_frac 0.5) completed (`nids_results.zip`): every C1 passed the gate, so R1 never ran on real data. See `notebooks/03_understand_our_model.ipynb`. Next: the pre-registered poison sweep (`experiments/prereg_sweep.md`, `notebooks/04_poison_sweep.ipynb`), not yet run.** Nothing here
is a result about the proposed method; it is data plumbing + an untrained model.

## Easiest: run `notebooks/00_audit.ipynb` -> `01_splits_experiences.ipynb` -> `02_train_evaluate.ipynb` on Kaggle (they call the scripts below).
Local plumbing check (synthetic, needs torch): `python smoke_test.py --work /tmp/nids_smoke`.

## Run order on Kaggle (Notebook, GPU T4)
1. Add the CICDDoS2019 dataset to the notebook (Add Input). Upload this folder as a
   Kaggle Dataset (or `git clone` your repo if Internet is on).
2. Cells:
```
%cd /kaggle/input/<your-code-dataset>/nids      # or your cloned repo
!python s0_audit.py --input /kaggle/input --hash                # Stage 0 (slow on 20+ GB; add --max-rows-per-file 2000000 for a first look)
!python s1_preprocess.py --input /kaggle/input --cap 100000 --split time   # Stage 1
!python s2_model.py                                              # Stage 2 self-test
```
Outputs go to `/kaggle/working/nids_out/`: `manifest.json`, `split_manifest.json`,
`scaler.json`, `class_map.json`, `{train,calib,val,test}.npz` (+ `_meta.csv`).

## What each stage does
| Stage | File | Purpose |
|---|---|---|
| 0 | `s0_audit.py` | Per-file schema, row/label counts, inf/NaN, duplicates, time range, hashes, warnings. Read the warnings before Stage 1. |
| 1 | `s1_preprocess.py` | One pass: global dedupe on model features, drop label conflicts, keep all distinct benign, bottom-k cap of distinct attack rows, split per class, refuse split overlap; saves UNSCALED parquet. |
| 1b | `s1b_experiences.py` | CI/CII row-id manifests (CII benign disjoint) + scaler/vocab fit on E0 train only; refuses thin classes and post-scaling train/protected overlap. |
| 3 | `s3_continual.py` | E0 training (T.pt + A0.pt), arms clean / poisoned / poisoned_repair, gate on val, one teacher-weighted repair, checkpoints + rounds.json under `runs/<scenario>/<run>/`. |
| 4 | `s4_report.py` | attempts.csv, per_class_val.csv, final_test.csv, target_recall.png, per_class_E<t>.png |
| 2 | `s2_model.py` | Feature tokenizer (numeric: `x*W_j+b_j`; protocol: embedding), Transformer encoder, expandable head. |

## Decisions baked in (change deliberately, and record why)
- IDs/timestamp dropped from inputs; **ports dropped** (reflection ports leak the label). Flip `DROP_PORTS` in `common.py` only as a labelled ablation.
- Dedupe happens before splitting. Scaler = signed-log1p, then standardise on train only, clip at +-5.
- Class map: BENIGN=0, others alphabetical, fixed for the whole study.
- `--split time` orders each class by Timestamp (no future leakage within a class but classes occupy different time windows); `--split random` is a flow-level shuffle and can leak near-identical neighbours. Neither is a deployment-chronology claim.
- `--cap` truncates huge attack files; BENIGN is tiny so it is kept whole. Check the per-split counts table: a rare class with a few dozen validation flows cannot support a recall gate.

## Not yet done (next stages)
Stages 3-4 exist and passed the synthetic smoke test only; no real-data training has been run.
