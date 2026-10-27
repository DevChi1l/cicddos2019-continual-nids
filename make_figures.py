"""Results storyboard figures from COMMITTED files only (no Kaggle, GPU, raw CSV, zip; trains and scores nothing).

  python make_figures.py                      # write docs/figures/figN_*.png + figN_*_data.csv
  python make_figures.py --check              # rebuild in a temp folder; fail if any plotted-data CSV differs from
                                              # docs/figures/, or a plotted value disagrees with its committed table
  python make_figures.py --execute-notebook   # run notebooks/11_results_storyboard.ipynb end to end (nbclient)

Every figure plots saved values only: measured points with min-max bars over seeds, no interpolation or smoothing.
Every panel says whether it is validation or held-out test (or data provenance), the seeds, and the source files.
Each PNG has a `_data.csv` next to it with exactly the plotted numbers (the table view of the figure).
`notebooks/11_results_storyboard.ipynb` calls these functions; the logic lives here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "experiments" / "results"
FIG = HERE / "docs" / "figures"
NOTEBOOK = HERE / "notebooks" / "11_results_storyboard.ipynb"
TOL = 1e-9

# validated categorical slots (light mode, adjacent pairs pass CVD/normal-vision; aqua/yellow need labels -> the
# data CSV and value labels are the relief); a fixed role per arm everywhere
C_CLEAN, C_GATE, C_SOFT, C_RELABEL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OK_FILL, BAD_FILL = "#e8f1fb", "#e34948"  # status: accepted (neutral tint) / rejected (critical red + text label)
SEQ = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue steps 250 / 450 / 650

REQUIRED = [
    "notebooks/manifest.json",
    "experiments/results/01_data/split_manifest.json",
    "experiments/results/04_sweep/decisions.csv",
    "experiments/results/04_sweep/hypotheses.csv",
    "experiments/results/04_sweep/final_test.csv",
    "experiments/results/04_sweep/test_summary.csv",
    "experiments/results/06_fu2/test/test_recall.csv",
    "experiments/results/06_fu2/test/test_summary.csv",
    "experiments/results/04b_all12/final_test.csv",
    "experiments/results/05_fu1/test/test_recall.csv",
    "experiments/results/07_oracle/test/test_recall.csv",
    "experiments/results/07_oracle/test/test_median.csv",
    "experiments/results/08_rc/rc/pilot.json",
    "experiments/results/09_rcs/rcs/screen.json",
    "experiments/results/10_rf/rf/trajectory_val.csv",
    "experiments/results/10_rf/rf/summary.json",
    "experiments/results/10_rf/test/rf/test/test_trajectory.csv",
    "experiments/results/10_rf/test/test_summary.csv",
] + [f"experiments/results/08_rc/runs/cii/rc_rep6_r0.10_s{s}/clean/rounds.json" for s in (42, 43, 44)] + [
    f"experiments/results/09_rcs/runs/cii/rcs_{t}_r0.10_s{s}/clean/rounds.json" for t in ("rcs6", "rcs5") for s in (42, 43, 44)]


class MissingArtifact(SystemExit):
    pass


def provenance() -> pd.DataFrame:
    """Every committed input with its sha256; a clear error (not a traceback) if anything is missing."""
    rows, missing = [], []
    for rel in REQUIRED:
        p = HERE / rel
        if not p.exists():
            missing.append(rel)
            continue
        rows.append({"file": rel, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:12]})
    if missing:
        raise MissingArtifact("missing committed result files (run from a full clone of "
                              "https://github.com/DevChi1l/cicddos2019-continual-nids at or after the storyboard commit; "
                              "`git pull` if your copy is older):\n  " + "\n  ".join(missing))
    return pd.DataFrame(rows)


def jl(rel):
    return json.loads((HERE / rel).read_text())


def rc(rel):
    return pd.read_csv(HERE / rel)


# ---- plotting helpers ----------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "legend.frameon": False, "svg.hashsalt": "nids"})
    return plt


def tag(ax, kind: str, extra: str = ""):
    """Panel badge: 'VALIDATION', 'HELD-OUT TEST' or 'DATA' plus seeds; top-left, ink only."""
    ax.text(0.0, 1.02, f"{kind}{'  |  ' + extra if extra else ''}", transform=ax.transAxes, fontsize=7.5,
            color=INK2, fontweight="bold", va="bottom")


def finish(fig, title: str, sources: list[str], name: str, data: pd.DataFrame, out: Path):
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Source: " + "; ".join(sources), fontsize=6.5, color=INK2, ha="left", va="bottom", wrap=True)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, metadata={"Software": None})
    data.to_csv(out / f"{name}_data.csv", index=False, float_format="%.10g")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return data


def yerr(mean, lo, hi):
    """min-max bar lengths; clipped at 0 because the mean of identical seeds can exceed them by 1 ulp."""
    m, lo, hi = (np.asarray(v, dtype=float) for v in (mean, lo, hi))
    return [np.clip(m - lo, 0, None), np.clip(hi - m, 0, None)]


def errbar(ax, x, mean, lo, hi, color, label, marker="o", ms=6):
    ax.errorbar(x, mean, yerr=yerr(mean, lo, hi), fmt=marker,
                color=color, ms=ms, lw=1.2, capsize=3, label=label, mec=SURFACE, mew=1.0)


# ---- figure 1: data pipeline ------------------------------------------------------------------------------
def fig1_data(out=FIG):
    plt = _plt()
    a = jl("notebooks/manifest.json")
    s = jl("experiments/results/01_data/split_manifest.json")
    st = s["stats"]
    labels = sorted(a["label_totals"], key=lambda k: -a["label_totals"][k])
    d = pd.DataFrame({"label": labels,
                      "raw_rows": [a["label_totals"][k] for k in labels],
                      "distinct_finite": [st["distinct_rows_by_label"][k] for k in labels],
                      "kept_after_conflict_drop_and_cap": [st["kept_by_label"][k] for k in labels],
                      "train_rows": [s["counts_per_split"]["train"].get(k, 0) for k in labels]})
    fig, (ax, side) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [3.3, 1]})
    y = np.arange(len(labels))
    for i, (col, lab) in enumerate((("raw_rows", "raw rows (audit)"),
                                     ("distinct_finite", "distinct finite rows (global dedupe)"),
                                     ("kept_after_conflict_drop_and_cap", "kept: conflicts dropped, attacks capped at 60,000"))):
        ax.barh(y + (i - 1) * 0.27, d[col], height=0.25, color=SEQ[i], label=lab, edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("rows (log scale)")
    b = labels.index("BENIGN")
    ax.get_yticklabels()[b].set_fontweight("bold")
    ax.text(2.5e5, b, f"BENIGN: {a['label_totals']['BENIGN']:,} raw = "
            f"{a['label_totals']['BENIGN'] / sum(a['label_totals'].values()):.2%} of all rows;\n"
            f"{st['kept_by_label']['BENIGN']:,} distinct kept, never capped", fontsize=8, color=INK, va="center")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, fontsize=7.5)
    tot = sum(a["label_totals"].values())
    side.axis("off")
    side.text(0.0, 0.95, "\n".join([
        f"{len(a['files'])} CSV files",
        f"{tot:,} rows read once",
        "",
        f"{sum(f['exact_duplicate_rows'] for f in a['files']):,}",
        "within-file exact duplicates",
        "",
        f"{st['n_conflict_vectors']:,}",
        "label-conflict vectors dropped",
        "",
        "Training used capped subsets only:",
        "98,833 rows (four-label)",
        "159,595 rows (12-label oracle, max)",
        "",
        "Filenames are not labels:",
        "03-11/UDPLag.csv holds 606,749 Syn",
        "and only 1,873 UDPLag rows"]), transform=side.transAxes, va="top", fontsize=8.5, color=INK, linespacing=1.35)
    tag(ax, "DATA PROVENANCE (no model)", "Stage 0 audit + Stage 1, seed 42")
    ax.title.set_visible(False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    return finish(fig, "1. From 70.4M raw rows to deduplicated, capped classes (14 raw labels)",
                  ["notebooks/manifest.json", "experiments/results/01_data/split_manifest.json"],
                  "fig1_data_pipeline", d, out)


# ---- figure 2: four-class sweep --------------------------------------------------------------------------
ARM_STYLE = {"clean": (C_CLEAN, "clean", "o"), "poisoned": (C_GATE, "poisoned, gate only", "s"),
             "poisoned_repair": (C_SOFT, "poisoned + soft repair (registered)", "^"),
             "poisoned_relabel": (C_RELABEL, "poisoned + teacher relabel (FU2, post hoc)", "D")}


def fig2_sweep(out=FIG):
    plt = _plt()
    dec = rc("experiments/results/04_sweep/decisions.csv")
    dec = dec[dec["run"].str.startswith("sweep_")]
    c1 = dec[(dec["arm"] == "poisoned") & (dec["experience"] == 1) & (dec["attempt"] == "C1")]
    rej = c1.groupby("poison_frac")["accepted"].agg(lambda s: int((~s.astype(bool)).sum())).rename("seeds_C1_rejected_E1")
    r1 = dec[dec["attempt"] == "R1"].groupby("poison_frac")["accepted"].agg(["size", "sum"]).rename(
        columns={"size": "soft_R1_ran", "sum": "soft_R1_accepted"})
    val = pd.concat([rej, r1], axis=1).fillna(0).astype(int).reset_index()
    ft = rc("experiments/results/04_sweep/final_test.csv")
    ft = ft[ft["run"].str.startswith("sweep_")]
    rows = []
    for (f, arm), g in ft.groupby(["poison_frac", "arm"]):
        for k in ("Syn", "LDAP", "UDP"):
            v = g[f"recall_{k}"]
            rows.append({"poison_frac": f, "arm": arm, "class": k, "mean": v.mean(), "min": v.min(), "max": v.max(),
                         "n_seeds": len(v)})
    test = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    ax = axes[0, 0]
    fs = val["poison_frac"].to_numpy()
    xs = np.arange(len(fs))
    ax.bar(xs - 0.18, val["seeds_C1_rejected_E1"], 0.34, color=C_GATE, label="poisoned C1 rejected at E1 (of 3 seeds)",
           edgecolor=SURFACE)
    ax.bar(xs + 0.18, val["soft_R1_accepted"], 0.34, color=C_SOFT, label="soft R1 accepted (E1+E2 attempts)",
           edgecolor=SURFACE, hatch="//")
    for i, r in enumerate(val.itertuples(index=False)):
        ax.text(xs[i] - 0.18, r.seeds_C1_rejected_E1 + 0.1, f"{int(r.seeds_C1_rejected_E1)}/3", ha="center", fontsize=8)
        ax.text(xs[i] + 0.18, r.soft_R1_accepted + 0.1, f"{int(r.soft_R1_accepted)}/{int(r.soft_R1_ran)}", ha="center",
                fontsize=8)
    ax.set_xticks(xs, [f"f = {f:g}" for f in fs])
    ax.set_ylabel("count")
    ax.set_ylim(0, 7.2)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Gate decisions vs share of stored Syn labels flipped", fontsize=9.5, loc="left", pad=16)
    tag(ax, "VALIDATION", "seeds 42-44; label = accepted/ran")
    for ax, k in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ("Syn", "LDAP", "UDP")):
        for j, arm in enumerate(("clean", "poisoned", "poisoned_repair")):
            g = test[(test["class"] == k) & (test["arm"] == arm)].sort_values("poison_frac")
            col, lab, mk = ARM_STYLE[arm]
            errbar(ax, np.arange(len(g)) + (j - 1) * 0.18, g["mean"], g["min"], g["max"], col, lab, mk)
            seeds = ft[ft["arm"] == arm].sort_values(["poison_frac", "seed"])
            fi = {f: i for i, f in enumerate(fs)}
            ax.scatter([fi[f] + (j - 1) * 0.18 + 0.07 for f in seeds["poison_frac"]], seeds[f"recall_{k}"], s=9,
                       facecolors="none", edgecolors=col, lw=0.8, zorder=3)
        ax.set_xticks(np.arange(len(fs)), [f"f = {f:g}" for f in fs])
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel(f"{k} recall (deployed model after E2)")
        ax.set_title(f"{k}: mean, bars = min-max over seeds", fontsize=9.5, loc="left", pad=16)
        tag(ax, "HELD-OUT TEST", "seeds 42-44, 46,527 rows, scored once")
    axes[0, 1].legend(loc="lower left", fontsize=7.5)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.text(0.52, 0.42, "gate only, f >= 0.95: learned in 0/3 seeds\nsoft repair, f = 1.0: learned in 0/3 seeds",
                transform=ax.transAxes, fontsize=7.5, color=INK2)
    axes[0, 1].text(0.02, 0.32, "hollow dots = individual seeds", transform=axes[0, 1].transAxes, fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    data = pd.concat([val.assign(panel="validation"), test.assign(panel="test")], ignore_index=True)
    return finish(fig, "2. Pre-registered four-class sweep (CII): the gate protects Syn; soft repair restores learning up to f = 0.95",
                  ["experiments/results/04_sweep/decisions.csv", "experiments/results/04_sweep/final_test.csv"],
                  "fig2_sweep", data, out)


# ---- figure 3: FU2 ------------------------------------------------------------------------------------------
def fig3_fu2(out=FIG):
    plt = _plt()
    tr = rc("experiments/results/06_fu2/test/test_recall.csv")
    arms = ["clean", "poisoned", "poisoned_repair", "poisoned_relabel"]
    rows = []
    for arm in arms:
        g = tr[tr["arm"] == arm]
        for k in ("Syn", "LDAP", "UDP", "benign_far"):
            rows.append({"arm": arm, "metric": k, "mean": g[k].mean(), "min": g[k].min(), "max": g[k].max(),
                         "n_seeds": len(g)})
    d = pd.DataFrame(rows)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 1.25]})
    x = np.arange(3)
    w = 0.2
    for j, arm in enumerate(arms):
        g = d[(d["arm"] == arm) & (d["metric"] != "benign_far")]
        col, lab, _ = ARM_STYLE[arm]
        hatch = "//" if arm in ("poisoned", "poisoned_repair") else None
        ax.bar(x + (j - 1.5) * w, g["mean"], w * 0.92, color=col, label=lab, edgecolor=SURFACE, hatch=hatch)
        ax.errorbar(x + (j - 1.5) * w, g["mean"], yerr=yerr(g["mean"], g["min"], g["max"]), fmt="none",
                    ecolor=INK, lw=0.8, capsize=2)
        for xi, m in zip(x + (j - 1.5) * w, g["mean"]):
            if m == 0:
                ax.text(xi, 0.015, "0", ha="center", fontsize=7, color=INK)
            else:
                ax.text(xi, m - 0.03, f"{m:.3f}", ha="center", va="top", fontsize=6.5, rotation=90,
                        color=INK if col == C_RELABEL else "white", fontweight="bold")
    ax.set_xticks(x, ["Syn (attacked, old)", "LDAP (new at E1)", "UDP (new at E2)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall of the deployed model after E2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=7.5)
    ax.set_title("All 2,000 stored Syn labels flipped to BENIGN before E1 (f = 1.0)", fontsize=9.5, loc="left", pad=16)
    tag(ax, "HELD-OUT TEST", "mean, bars = min-max, seeds 42-44")
    g = d[d["metric"] == "benign_far"]
    for j, arm in enumerate(arms):
        r = g[g["arm"] == arm].iloc[0]
        col, _, _ = ARM_STYLE[arm]
        ax2.bar(j, r["mean"], 0.7, color=col, edgecolor=SURFACE, hatch="//" if arm in ("poisoned", "poisoned_repair") else None)
        ax2.errorbar(j, r["mean"], yerr=yerr([r["mean"]], [r["min"]], [r["max"]]), fmt="none", ecolor=INK, lw=0.8,
                     capsize=2)
        ax2.text(j, r["max"] + 0.00008, f"{r['mean']:.4f}", ha="center", fontsize=7)
    ax2.set_xticks(range(4), ["clean", "gate\nonly", "soft\nrepair", "relabel\n(FU2)"])
    ax2.set_ylabel("benign false-alarm rate (19,028 benign rows)")
    ax2.set_ylim(0, 0.0026)
    ax2.set_title("Benign FAR", fontsize=9.5, loc="left", pad=16)
    tag(ax2, "HELD-OUT TEST")
    ax2.text(0.5, 0.80, "FAR 0 for gate-only and soft\nrepair is not safety: the stale\nA0 calls LDAP/UDP benign",
             transform=ax2.transAxes, fontsize=7, color=INK2, ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return finish(fig, "3. PRIMARY RESULT: teacher relabelling (FU2, POST HOC method) learns all four classes at 100% flips",
                  ["experiments/results/06_fu2/test/test_recall.csv (FU2 relabel + sweep f = 1.0 controls, same rows)"],
                  "fig3_fu2_defence", d, out)


# ---- figure 4: 12-label continual vs joint oracle ---------------------------------------------------------------
CLASSES12 = ["BENIGN", "Syn", "DNS", "LDAP", "MSSQL", "NTP", "NetBIOS", "SNMP", "SSDP", "TFTP", "UDP", "UDPLag"]


def fig4_oracle(out=FIG):
    plt = _plt()
    s3 = rc("experiments/results/04b_all12/final_test.csv")
    s3 = s3[(s3["scenario"] == "cii") & (s3["arm"] == "clean")].iloc[0]
    fu1 = rc("experiments/results/05_fu1/test/test_recall.csv")
    fu1 = fu1[fu1["scenario"] == "cii"].set_index("class")["fu1"]
    orc = rc("experiments/results/07_oracle/test/test_recall.csv")
    orc = orc[orc["scenario"] == "cii"]
    rows = []
    for k in CLASSES12:
        rows.append({"class": k, "model": "continual s3 clean (seed 42)", "mean": s3[f"test_recall_{k}"],
                     "min": s3[f"test_recall_{k}"], "max": s3[f"test_recall_{k}"]})
        rows.append({"class": k, "model": "continual FU1 retry (seed 42)", "mean": fu1[k], "min": fu1[k], "max": fu1[k]})
        for setting in ("e10", "e20"):
            v = orc[orc["setting"] == setting][k]
            rows.append({"class": k, "model": f"joint oracle {setting} (median, seeds 42-44)", "mean": v.median(),
                         "min": v.min(), "max": v.max()})
    d = pd.DataFrame(rows)
    models = list(dict.fromkeys(d["model"]))
    cols = [C_CLEAN, C_GATE, C_SOFT, C_RELABEL]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [4, 1.2]})
    x = np.arange(len(CLASSES12))
    w = 0.2
    for j, (m, col) in enumerate(zip(models, cols)):
        g = d[d["model"] == m].set_index("class").loc[CLASSES12]
        ax.bar(x + (j - 1.5) * w, g["mean"], w * 0.92, color=col, label=m, edgecolor=SURFACE,
               hatch=".." if m.startswith("joint") else None)
        if m.startswith("joint"):  # continual runs are single-seed: no range to draw
            ax.errorbar(x + (j - 1.5) * w, g["mean"], yerr=yerr(g["mean"], g["min"], g["max"]), fmt="none",
                        ecolor=INK, lw=0.7, capsize=1.5)
    ax.set_xticks(x, CLASSES12, rotation=0, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("per-class recall")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.08), ncol=2, fontsize=7.5)
    ax.set_title("12 labels, CII, 123,281 test rows (19,028 benign)", fontsize=9.5, loc="left", pad=16)
    tag(ax, "HELD-OUT TEST", "continual: one seed; oracle: median, bars = min-max over 3 seeds")
    cnt = d.assign(learned=d["mean"] > 0, at80=d["mean"] >= 0.80).groupby("model", sort=False)[["learned", "at80"]].sum()
    yy = np.arange(len(models))
    ax2.barh(yy + 0.18, cnt["learned"], 0.34, color=[cols[i] for i in range(len(models))], edgecolor=SURFACE)
    ax2.barh(yy - 0.18, cnt["at80"], 0.34, color=[cols[i] for i in range(len(models))], edgecolor=SURFACE, hatch="//",
             alpha=0.75)
    for i, m in enumerate(models):
        ax2.text(cnt.loc[m, "learned"] + 0.2, i + 0.18, f"{cnt.loc[m, 'learned']} with recall > 0", va="center", fontsize=7)
        ax2.text(cnt.loc[m, "at80"] + 0.2, i - 0.18, f"{cnt.loc[m, 'at80']} with recall >= 0.80", va="center", fontsize=7)
    ax2.set_yticks(yy, ["s3 clean", "FU1", "oracle e10", "oracle e20"])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 17)
    ax2.set_xlabel("classes of 12")
    ax2.set_title("Retained classes", fontsize=9.5, loc="left", pad=16)
    tag(ax2, "HELD-OUT TEST")
    fig.text(0.01, 0.045, "The joint oracle (dotted bars) is ONE model trained on all 12 labels at once: a diagnostic "
             "reference with no updates and no gate, not a defended continual result. The 12-label continual goal is unsolved.",
             fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    data = pd.concat([d, cnt.reset_index().rename(columns={"learned": "classes_recall_gt0", "at80": "classes_recall_ge_0.80"})],
                     ignore_index=True)
    return finish(fig, "4. LIMITATION: continual 12-label runs (CII) retain 6 of 12 classes; a joint oracle is not a continual result",
                  ["experiments/results/04b_all12/final_test.csv", "experiments/results/05_fu1/test/test_recall.csv",
                   "experiments/results/07_oracle/test/test_recall.csv"], "fig4_12label_vs_oracle", data, out)


# ---- figure 5: RC / RCS clean pilots ----------------------------------------------------------------------------
def fig5_pilots(out=FIG):
    plt = _plt()
    sched = [("RC  (rep6)", "experiments/results/08_rc/runs/cii/rc_rep6_r0.10_s{}/clean/rounds.json"),
             ("RCS rcs6", "experiments/results/09_rcs/runs/cii/rcs_rcs6_r0.10_s{}/clean/rounds.json"),
             ("RCS rcs5", "experiments/results/09_rcs/runs/cii/rcs_rcs5_r0.10_s{}/clean/rounds.json")]
    rows = []
    for name, pat in sched:
        for seed in (42, 43, 44):
            for r in jl(pat.format(seed)):
                v, ref = r["val"], r["ref_val"]
                worst = min((k for k in r["old_classes"] if k != "BENIGN"),
                            key=lambda k: v["recall"].get(k, 0) - (ref["recall"][k] - 0.05))
                rows.append({"schedule": name, "seed": seed, "t": r["experience"], "new_class": r["new_classes"][0],
                             "accepted": bool(r["accepted"]), "worst_old_class": worst,
                             "worst_old_recall": v["recall"].get(worst, 0.0),
                             "worst_old_floor": ref["recall"][worst] - 0.05,
                             "Syn_recall": v["recall"]["Syn"], "reasons": "; ".join(r["gate_reasons"])})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), gridspec_kw={"width_ratios": [6, 6, 5]})
    for ax, (name, _) in zip(axes, sched):
        g = d[d["schedule"] == name]
        ts = sorted(g["t"].unique())
        for i, seed in enumerate((42, 43, 44)):
            for t in ts:
                r = g[(g["seed"] == seed) & (g["t"] == t)].iloc[0]
                ax.add_patch(plt.Rectangle((t - 0.47, i - 0.45), 0.94, 0.9, color=OK_FILL if r["accepted"] else BAD_FILL,
                                           lw=0))
                txt = ("accepted" if r["accepted"] else
                       f"REJECTED\nold {r['worst_old_class']}\n{r['worst_old_recall']:.3f}\n< {r['worst_old_floor']:.3f}")
                ax.text(t, i, txt, ha="center", va="center", fontsize=6, color=INK if r["accepted"] else "white",
                        fontweight="normal" if r["accepted"] else "bold")
        ax.set_xlim(0.5, max(ts) + 0.5)
        ax.set_ylim(2.6, -0.6)
        ax.set_xticks(ts, [f"E{t}\n+{g[g['t'] == t]['new_class'].iloc[0]}" for t in ts], fontsize=7.5)
        ax.set_yticks(range(3), ["seed 42", "seed 43", "seed 44"])
        ax.grid(False)
        nrej = int((~g["accepted"]).sum())
        ax.set_title(f"{name}: {nrej} rejected updates", fontsize=9.5, loc="left", pad=16)
        tag(ax, "VALIDATION", "clean arm only, no poison")
    fig.text(0.01, 0.075, "Each cell is one clean C1 candidate; a rejected cell names the old class with the smallest gate "
             "margin: its recall < floor (floor = last accepted recall - 0.05).\nEvery rejection is an old NON-Syn class "
             "forgotten when an overlapping class arrives; Syn validation recall of every candidate stayed "
             f"{d['Syn_recall'].min():.4f}-{d['Syn_recall'].max():.4f}.\nThe registered rules required a clean sequence that "
             "passes the gate first, so both experiments STOPPED: no poisoned arm was trained and no test exists.",
             fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    return finish(fig, "5. RC and RCS (pre-registered): clean pilots failed, so the poisoned runs never started",
                  ["experiments/results/08_rc/runs/.../clean/rounds.json", "experiments/results/09_rcs/runs/.../clean/rounds.json",
                   "08_rc/rc/pilot.json", "09_rcs/rcs/screen.json"], "fig5_rc_rcs_pilots", d, out)


# ---- figure 6: RF trajectories -------------------------------------------------------------------------------------
RF_SERIES = [("clean", 0.10, "clean (no flips)", C_CLEAN, "o"), ("poisoned", 0.05, "poisoned, 5% per update", C_SOFT, "^"),
             ("poisoned", 0.10, "poisoned, 10% per update", C_GATE, "s")]


def _rf_points(df, split):
    rows = []
    for arm, rho, lab, _, _ in RF_SERIES:
        g = df[(df["arm"] == arm) & (np.isclose(df["rho"], rho))]
        for t, h in g.groupby("t"):
            for m in ("Syn_recall", "Syn_mean_p"):
                rows.append({"split": split, "series": lab, "arm": arm, "rho": rho, "t": int(t), "metric": m,
                             "mean": h[m].mean(), "min": h[m].min(), "max": h[m].max(), "n_seeds": len(h)})
    return pd.DataFrame(rows)


def fig6_rf(out=FIG):
    plt = _plt()
    val = rc("experiments/results/10_rf/rf/trajectory_val.csv")
    test = rc("experiments/results/10_rf/test/rf/test/test_trajectory.csv")
    d = pd.concat([_rf_points(val, "validation"), _rf_points(test, "test")], ignore_index=True)
    po = val[(val["arm"] == "poisoned") & (val["t"] > 0)]
    rep = val[(val["arm"] == "poisoned_repair") & (val["t"] > 0)]
    acc, r1 = int(po["c1_accepted"].sum()), int(rep["r1_ran"].sum())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for row, split in enumerate(("validation", "test")):
        for colj, (m, ylab) in enumerate((("Syn_recall", "Syn recall"), ("Syn_mean_p", "mean p(Syn | Syn rows)"))):
            ax = axes[row, colj]
            for j, (arm, rho, lab, col, mk) in enumerate(RF_SERIES):
                g = d[(d["split"] == split) & (d["series"] == lab) & (d["metric"] == m)].sort_values("t")
                xs = g["t"] + (j - 1) * 0.12
                ax.plot(xs, g["mean"], color=col, lw=1.0, alpha=0.6)
                errbar(ax, xs, g["mean"], g["min"], g["max"], col, lab, mk, ms=5)
            ax.set_ylabel(ylab + " (axis truncated)")
            ax.set_ylim(0.97, 1.0005) if m == "Syn_recall" else ax.set_ylim(0.7, 1.01)
            tag(ax, "VALIDATION" if split == "validation" else "HELD-OUT TEST (scored once)", "seeds 42-44, bars = min-max")
            if row == 1:
                ax.set_xlabel("update t (A_t after t refresh updates; 0 = A0)")
            ax.set_xticks(range(7))
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    axes[0, 0].text(0.98, 0.06, f"poisoned arm: C1 accepted {acc}/{len(po)} updates\n(both rates x 3 seeds x 6 updates)\n"
                    f"repair arm: R1 ran {r1} times in {len(rep)} updates\n-> repair never triggered",
                    transform=axes[0, 0].transAxes, ha="right", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
    axes[1, 1].text(0.02, 0.06, "10% per update: 60% of stored Syn labels flipped by t = 6.\nRecall holds; confidence falls; "
                    "the recall-based gate never reacts.", transform=axes[1, 1].transAxes, fontsize=7.5, color=INK)
    fig.text(0.01, 0.035, "Lines only join the measured update points (no interpolation). Fixed classes BENIGN, Syn, TFTP; "
             "six refresh updates, no new class. POST HOC study; the confidence loss was NOT repaired.", fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return finish(fig, "6. LIMITATION (RF, post hoc): repeated Syn flips lower confidence while recall and the gate do not react",
                  ["experiments/results/10_rf/rf/trajectory_val.csv", "experiments/results/10_rf/test/rf/test/test_trajectory.csv"],
                  "fig6_rf_trajectory", d, out)


# ---- figure 7: experiment status -----------------------------------------------------------------------------------------
STATUS = [  # experiment, labels, pre-registered, post hoc, validation, held-out test, outcome, evidence folder
    ("main (02)", "4", "no", "no", "yes", "yes*", "completed", "02_main"),
    ("sweep (04A)", "4", "yes", "no", "yes", "yes*", "completed", "04_sweep"),
    ("all12 (04B)", "12", "no", "no", "yes", "yes*", "completed, exploratory", "04b_all12"),
    ("FU1 (05)", "12", "yes", "no", "yes", "yes", "completed, negative", "05_fu1"),
    ("FU2 (06)", "4", "yes", "YES", "yes", "yes", "completed, PRIMARY", "06_fu2"),
    ("oracle (07)", "12", "yes", "no", "yes", "yes", "completed, diagnostic", "07_oracle"),
    ("RC (08)", "7", "yes", "no", "clean pilot", "no", "STOPPED", "08_rc"),
    ("RCS (09)", "6-7", "yes", "no", "clean screen", "no", "STOPPED", "09_rcs"),
    ("RF (10)", "3", "yes", "YES", "yes", "yes", "completed, limitation", "10_rf"),
]


def fig7_status(out=FIG):
    plt = _plt()
    cols = ["experiment", "labels", "pre-registered", "post hoc", "validation", "held-out test", "outcome", "evidence"]
    d = pd.DataFrame(STATUS, columns=cols)
    for r in d.itertuples():  # every claimed artifact must exist in git
        base = RES / r.evidence
        assert base.is_dir(), f"missing evidence folder {base}"
        has_test = any(p.name in ("final_test.json", "final_test.csv", "test.json", "test_trajectory.csv")
                       for p in base.rglob("*"))
        assert (r[6] != "no") == has_test, f"{r.experiment}: test status disagrees with the files in {base}"
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    w = [0.12, 0.06, 0.11, 0.15, 0.11, 0.1, 0.2, 0.1]
    xs = np.cumsum([0] + w[:-1]) + 0.01
    for x, c in zip(xs, cols):
        ax.text(x, 0.95, c, fontsize=8.5, fontweight="bold", color=INK, va="center")
    for i, r in enumerate(d.itertuples(index=False)):
        y = 0.85 - i * 0.095
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.045), 1, 0.09, color="#f1f0ec", lw=0, transform=ax.transAxes))
        for x, v, c in zip(xs, r, cols):
            txt, color, weight = str(v), INK, "normal"
            if c in ("pre-registered", "post hoc", "validation", "held-out test"):
                txt = {"yes": "✓ yes", "no": "– no", "YES": "✓ YES (disclosed)", "yes*": "✓ yes*"}.get(v, f"◐ {v}")
            if c == "outcome":
                weight = "bold" if v in ("STOPPED", "completed, PRIMARY", "completed, limitation") else "normal"
                color = BAD_FILL if v == "STOPPED" else INK
                txt = ("■ " if v == "STOPPED" else "● ") + v
            ax.text(x, y, txt, fontsize=8, color=color, fontweight=weight, va="center")
    ax.text(0.01, -0.02, "* test was computed inside the run, before any decisions commit (disclosed as sweep deviation D1; "
            "the same holds for main and all12); never displayed or used for a decision.\nFU1 onward: validation committed "
            "first, then test scored once by a commit-checked command.", fontsize=7.2, color=INK2, transform=ax.transAxes,
            va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return finish(fig, "7. Experiment status: what was registered, post hoc, validated, held-out tested or stopped",
                  ["experiments/results/*/README.md", "experiments/prereg_*.md (status asserted against the files present)"],
                  "fig7_status", d, out)


FIGURES = [fig1_data, fig2_sweep, fig3_fu2, fig4_oracle, fig5_pilots, fig6_rf, fig7_status]


# ---- plotted numbers vs committed tables (independent read) --------------------------------------------------------------
def verify(data: dict) -> list[str]:
    msgs = []
    d = data["fig1_data"]
    sp = jl("experiments/results/01_data/split_manifest.json")["stats"]
    kept = d.set_index("label")["kept_after_conflict_drop_and_cap"]
    assert kept.to_dict() == sp["kept_by_label"], "fig1 kept counts differ from split_manifest.json"
    raw = jl("notebooks/manifest.json")["label_totals"]
    assert d.set_index("label")["raw_rows"].to_dict() == raw, "fig1 raw counts differ from the audit manifest"
    msgs.append("fig1: kept counts == 01_data/split_manifest.json; raw == notebooks/manifest.json")

    d = data["fig2_sweep"]
    ts = rc("experiments/results/04_sweep/test_summary.csv").set_index(["poison_frac", "arm"])
    t = d[d["panel"] == "test"]
    for _, r in t.iterrows():
        for s in ("mean", "min", "max"):
            assert abs(r[s] - ts.loc[(r["poison_frac"], r["arm"]), f"recall_{r['class']}_{s}"]) < TOL, (r, s)
    h = rc("experiments/results/04_sweep/hypotheses.csv")
    hv = d[d["panel"] == "validation"].set_index("poison_frac")["seeds_C1_rejected_E1"]
    for _, r in h[h["hypothesis"].isin(["H1", "H2"])].iterrows():
        f = float(r["claim"].split(":")[0].split("=")[1])
        k = int(r["evidence"].split("/")[0])
        assert hv[f] == (3 - k if r["hypothesis"] == "H1" else k), (f, r["evidence"])
    msgs.append("fig2: test points == 04_sweep/test_summary.csv (mean/min/max); rejections == hypotheses.csv H1/H2")

    d = data["fig3_fu2"]
    ts = rc("experiments/results/06_fu2/test/test_summary.csv").set_index("arm")
    for _, r in d.iterrows():
        for s in ("mean", "min", "max"):
            assert abs(r[s] - ts.loc[r["arm"], f"{r['metric']}_{s}"]) < TOL, (r, s)
    msgs.append("fig3: every bar and range == 06_fu2/test/test_summary.csv")

    d = data["fig4_oracle"].dropna(subset=["class"])
    med = rc("experiments/results/07_oracle/test/test_median.csv")
    med = med[med["scenario"].str.startswith("cii")].set_index("setting")
    vs = rc("experiments/results/07_oracle/test/test_vs_continual.csv")
    vs = vs[vs["scenario"] == "cii"].set_index("metric")
    for _, r in d.iterrows():
        if r["model"].startswith("joint"):
            assert abs(r["mean"] - med.loc[r["model"].split()[2], r["class"]]) < TOL, r
        else:
            col = "continual_s3_clean" if "s3" in r["model"] else "continual_fu1"
            assert abs(r["mean"] - vs.loc[f"recall_{r['class']}", col]) < TOL, r
    msgs.append("fig4: oracle medians == 07_oracle/test/test_median.csv; continual == test_vs_continual.csv")

    d = data["fig5_pilots"]
    p = jl("experiments/results/08_rc/rc/pilot.json")["attempts"][0]["rejections_per_class"]
    g = d[(d["schedule"] == "RC  (rep6)") & (~d["accepted"])].groupby("new_class").size()
    assert {k: v for k, v in p.items() if v} == g.to_dict()
    scr = jl("experiments/results/09_rcs/rcs/screen.json")
    for tag_, c in scr["candidates"].items():
        acc = d[d["schedule"] == f"RCS {tag_}"].groupby("seed")["accepted"].all()
        assert [bool(acc[int(s)]) for s in c["per_seed"]] == [v["all_accepted"] for v in c["per_seed"].values()]
    msgs.append("fig5: rejected cells == 08_rc/rc/pilot.json and 09_rcs/rcs/screen.json")

    d = data["fig6_rf"]
    ts = rc("experiments/results/10_rf/test/test_summary.csv")
    for _, r in d[(d["split"] == "test") & (d["t"].isin([0, 6]))].iterrows():
        row = ts[(np.isclose(ts["rho"], r["rho"])) & (ts["arm"] == r["arm"]) & (ts["model"] == f"A{r['t']}")].iloc[0]
        for s in ("mean", "min", "max"):
            assert abs(r[s] - row[f"{r['metric']}_{s}"]) < TOL, (r, s)
    pv = jl("experiments/results/10_rf/rf/summary.json")["predictions_val"]
    for rho in (0.10, 0.05):
        per = pv[f"rho_{rho:.2f}"]["per_seed"]
        g = d[(d["split"] == "validation") & (d["arm"] == "poisoned") & np.isclose(d["rho"], rho) &
              (d["metric"] == "Syn_mean_p") & (d["t"] == 6)].iloc[0]
        assert abs(g["mean"] - np.mean([v["poisoned_mean_p_AT"] for v in per.values()])) < TOL
    msgs.append("fig6: test A0/A6 == 10_rf/test/test_summary.csv; validation A6 mean p == rf/summary.json")
    msgs.append("fig7: every status cell asserted against the evidence folders present in git")
    return msgs


def build(out: Path) -> dict:
    provenance()
    data = {}
    for f in FIGURES:
        name = f.__name__
        data[name] = f(out)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--execute-notebook", action="store_true")
    a = ap.parse_args()
    try:
        if a.execute_notebook:
            import os
            import nbformat
            from nbclient import NotebookClient
            nb = nbformat.read(NOTEBOOK, as_version=4)
            with tempfile.TemporaryDirectory() as kdir:  # a throw-away kernel spec for THIS interpreter (no user setup)
                spec = Path(kdir) / "kernels" / "nids-storyboard"
                spec.mkdir(parents=True)
                (spec / "kernel.json").write_text(json.dumps({
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "nids storyboard", "language": "python"}))
                os.environ["JUPYTER_PATH"] = kdir + os.pathsep + os.environ.get("JUPYTER_PATH", "")
                NotebookClient(nb, timeout=600, kernel_name="nids-storyboard",
                               resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
            n = sum(c.cell_type == "code" for c in nb.cells)
            errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
            assert not errs, errs
            print(f"OK: {NOTEBOOK.relative_to(HERE)} executed end to end ({n} code cells, no errors)")
            return
        if a.check:
            with tempfile.TemporaryDirectory() as tmp:
                data = build(Path(tmp))
                bad = [p.name for p in sorted(Path(tmp).glob("*_data.csv"))
                       if not (FIG / p.name).exists() or (FIG / p.name).read_bytes() != p.read_bytes()]
                missing_png = [f"{f.__name__}" for f in FIGURES if not list(FIG.glob(f"{f.__name__.split('_')[0]}_*.png"))]
            msgs = verify(data)
            if bad or missing_png:
                print("FAIL: plotted data differs from docs/figures/ or figures missing:", bad + missing_png)
                sys.exit(1)
            print("\n".join(f"- {m}" for m in msgs))
            print(f"OK: {len(FIGURES)} figures rebuilt from committed files; plotted numbers == docs/figures/*_data.csv "
                  "and == the committed result tables")
            return
        data = build(FIG)
        for m in verify(data):
            print("-", m)
        print(f"wrote {len(FIGURES)} figures + data CSVs to {FIG.relative_to(HERE)}")
    except MissingArtifact as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
