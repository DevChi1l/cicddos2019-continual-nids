# Dashboard (offline, one file)

`dashboard/index.html` opens in any browser with no server and no internet. It has three tabs:

1. **Detect a flow:** four deployed models (f = 1.0, seed 42: clean, gate only, soft repair, relabel repair) run in
   the browser on real held-out test flows. You can edit a flow and see each verdict change. On load, every model
   runs on all sampled flows in a background thread and is compared with PyTorch's own outputs. The header chip
   reports the match.
2. **Poison the buffer:** the committed gate decisions (validation) and deployed-model results (held-out test) of the
   pre-registered sweep and FU2, for any flip fraction, seed and defence.
3. **What we claim:** the claims ladder, the RF confidence blind spot, the data funnel, integrity facts and (once
   committed) the CG confidence-gate follow-up.

Build (reads committed files only; no torch needed):

```
python dashboard/build_dashboard.py --assets dashboard/dashboard_assets.json
python dashboard/build_dashboard.py --assets dashboard/dashboard_assets.json --check
```

`dashboard_assets.json` is written on Kaggle by `make_dashboard_assets.py` (kernel `kaggle/cg_val/`). It also asserts
that the exported models reproduce the committed full-test recalls exactly. `engine.js` is the model's forward pass
in plain JavaScript; it matches PyTorch to about 1e-7 in probability.
