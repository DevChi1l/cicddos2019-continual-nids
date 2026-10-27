# `main` run (notebook 02): saved JSON/CSV copies (no checkpoints, no buffers)

Run `runs/cii/main`, CII, seed 42, poison Syn -> BENIGN at f = 0.5 once before E1, schedule E0 Syn, E1 LDAP, E2 UDP.
Trained on Kaggle on 2026-10-20 (about 19:20 UTC), before the repository's first commit `244bc06`; the code is the
Stage 3 of that first commit with default flags. `repro_s42` in the sweep reproduced its gate decisions (H0).

Source: `nids_results.zip` (sha256 `3352f521…`, not committed). Copied unchanged: `config.json`,
`run_manifest.json` (teacher sha256 and `T_unchanged: true`), `A0_metrics.json`, each arm's `rounds.json` and
`final_test.json`, and `report/{attempts,final_test,per_class_val}.csv` written by `s4_report.py`.

**Status:** an early, single-seed run. Test was scored inside the run (no separate decisions commit existed yet);
no decision ever read test. Every C1 was accepted in all three arms, so R1 never ran: `main` does not test repair.
Its role is the reference that the pre-registered sweep reproduced (`repro_s42`, H0 supported).

Mean confidence figures for `main` quoted in `docs/slides_outline.md` (about 0.79 on true-Syn buffer rows, teacher
p 0.004 on flipped rows) come from notebook 03's read-only analysis (`s5_explain.py`), not from a file here.
