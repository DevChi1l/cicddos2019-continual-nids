# Dashboard (offline, one file)

## How to open

1. **Double-click `dashboard/index.html`.** It is one self-contained file and works offline, with no server and no
   internet (no CDN, no web fonts).
2. **`python dashboard/serve.py`** (standard library only) serves this folder on http://localhost:8000 and opens the
   browser. Use this if your browser blocks the page's background worker on `file://` pages.
3. **Online:** https://devchi1l.github.io/cicddos2019-continual-nids/dashboard/ . This needs GitHub Pages to be
   enabled for the repository (Settings → Pages → Deploy from a branch → `main`, `/ (root)`).

Keys: `1` `2` `3` switch tabs; `←` `→` previous/next flow and `R` random flow on tab 1; `P` presenter mode (larger
fonts, no footer); `?` lists the shortcuts. Ctrl+P prints all three tabs one after another as a handout.

`dashboard/index.html` opens in any browser with no server and no internet. It has six screens:

1. **Overview:** three headline numbers and the method in four steps. All numbers are computed from the committed files.
2. **Live detector:** four deployed models (f = 1.0, seed 42: clean, gate only, soft repair, relabel repair) run in
   the browser on real held-out test flows; edit a flow and watch each verdict. The *Batch check* view runs every
   model on the 1,200 sampled flows in a background thread and compares them with PyTorch's own outputs; the header
   chip reports the match.
3. **Try your own data:** drop a CSV of flows (CICDDoS2019 columns, raw headers accepted, optional `Label`). It shows a
   validation report, predictions of all four models, and with labels accuracy, per-class precision / recall / F1
   and a confusion matrix. Missing features are filled with the BENIGN median (scaled space) and reported. Runs
   entirely in the browser; nothing is uploaded. `dashboard/sample_flows.csv` (200 held-out test rows, 50 per class,
   written by the build) is the same sample as the *Load sample* button.
4. **Attack lab:** the committed gate decisions (validation) and deployed-model results (held-out test) of the
   pre-registered sweep and FU2, for any flip fraction, seed and defence; *Play* steps through 75-100%.
5. **Results:** one SVG chart per result (sweep, per-class recall and precision, accuracy vs macro recall, RF blind
   spot, confidence gate, five classes, and the SEEDS follow-up once committed), each with a computed takeaway.
6. **Evidence:** the claims ladder (seeds 42-44), data funnel, integrity facts and the follow-up tables.

Build (reads committed files only; no torch needed):

```
python dashboard/build_dashboard.py --assets dashboard/dashboard_assets.json
python dashboard/build_dashboard.py --assets dashboard/dashboard_assets.json --check
```

`dashboard_assets.json` is written on Kaggle by `make_dashboard_assets.py` (kernel `kaggle/cg_val/`). It also asserts
that the exported models reproduce the committed full-test recalls exactly. `engine.js` is the model's forward pass
in plain JavaScript; it matches PyTorch to about 1e-7 in probability.
