"""Stage 6 - Poison-fraction sweep tables, in two steps (reads saved JSON only, trains nothing).

  python s6_sweep.py --decisions   # VALIDATION ONLY: gate decisions + forced-R1 diagnostics
  python s6_sweep.py --test        # test metrics; refuses until sweep/decisions.csv exists

The pre-registration (experiments/prereg_sweep.md) asks for decisions.csv to be committed to git
before --test is run. --decisions never opens final_test.json; --test only adds test numbers.

Writes into <out>/sweep/:
  decisions.csv      one row per (run, arm, experience, attempt): accepted, Syn val recall vs floor,
                     benign FAR, mean p(Syn) on val Syn rows, teacher-weight stats
  diagnostics.csv    one row per forced R1 (DIAGNOSTIC, never a decision): would it pass the gate
  hypotheses.csv     H0-H6 of the pre-registration, evaluated from decisions/diagnostics only
  decisions.png      Syn val recall of the poisoned C1 at E1 against its gate floor, per f and seed
  final_test.csv     (--test) test metrics of each arm's final model, per run
  test_summary.csv   (--test) per f and arm: mean, min, max over seeds
  test_syn.png       (--test) final test Syn recall per arm against f
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import default_work_dir, load_json

ARM_COLOR = {"clean": "#2a78d6", "poisoned": "#eb6834", "poisoned_repair": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
NAME = re.compile(r"^sweep_f([0-9.]+)_s(\d+)$")


def runs_in(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern) if (p / "config.json").exists())


def decision_rows(rdir: Path) -> tuple[list[dict], list[dict]]:
    cfg = load_json(rdir / "config.json")
    args, target = cfg["args"], cfg["args"]["poison_target"]
    base = {"run": rdir.name, "poison_frac": args["poison_frac"], "seed": cfg.get("seed", 42)}
    dec, diag = [], []
    for arm in ARM_COLOR:
        if not (rdir / arm / "rounds.json").exists():
            continue
        for r in load_json(rdir / arm / "rounds.json"):
            v, ref, tw = r["val"], r["ref_val"], r["teacher_weights"] or {}
            dec.append({**base, "arm": arm, "experience": r["experience"], "attempt": r["attempt"],
                        "accepted": r["accepted"], "gate_reasons": "; ".join(r["gate_reasons"]),
                        f"{target}_val_recall": v["recall"].get(target),
                        f"{target}_floor": ref["recall"][target] - args["gate_old_drop"],
                        f"{target}_val_mean_p": v.get("mean_conf", {}).get(target),
                        f"{target}_ref_mean_p": ref.get("mean_conf", {}).get(target),
                        "benign_far": v["benign_far"],
                        "new_recall_min": min(v["recall"].get(k, 0) for k in r["new_classes"]),
                        "repair_method": tw.get("method", "soft" if r["attempt"] == "R1" else None),
                        "w_flipped": tw.get("mean_w_flipped"), "w_clean": tw.get("mean_w_clean_known")})
        if (rdir / arm / "r1_diag.json").exists():
            for r in load_json(rdir / arm / "r1_diag.json"):
                v, ref, tw = r["val"], r["ref_val"], r["teacher_weights"]
                diag.append({**base, "arm": arm, "experience": r["experience"], "method": r["method"],
                             "label": "DIAGNOSTIC (not a decision)", "would_pass_gate": r["would_pass_gate"],
                             "gate_reasons": "; ".join(r["gate_reasons"]),
                             f"{target}_val_recall": v["recall"].get(target),
                             f"{target}_floor": ref["recall"][target] - args["gate_old_drop"],
                             f"{target}_val_mean_p": v["mean_conf"].get(target),
                             "benign_far": v["benign_far"],
                             "w_flipped": tw.get("mean_w_flipped"), "w_clean": tw.get("mean_w_clean_known"),
                             "n_quarantined": tw.get("n_quarantined"),
                             "n_quarantined_flipped": tw.get("n_quarantined_flipped"),
                             "n_relabelled": tw.get("n_relabelled"),
                             "n_relabelled_to_true": tw.get("n_relabelled_to_true")})
    return dec, diag


def hypotheses(dec: pd.DataFrame, diag: pd.DataFrame, repro: dict | None) -> pd.DataFrame:
    """Pre-registered H0-H6, from validation decisions only."""
    sw = dec[dec["run"].str.match(NAME)]
    c1 = sw[(sw["arm"] == "poisoned") & (sw["experience"] == 1) & (sw["attempt"] == "C1")]
    rows = []

    def add(h, claim, n_ok, n, result):
        rows.append({"hypothesis": h, "claim": claim, "evidence": f"{n_ok}/{n}", "result": result})

    if repro is not None:
        add("H0", "repro_s42 makes the same gate decisions as main", int(repro["same"]), 1,
            "supported" if repro["same"] else "NOT supported")
    for fs, h, want, need in [((0.75, 0.9), "H1", True, "all"), ((0.95, 1.0), "H2", False, "2of3")]:
        for f in fs:
            s = c1[np.isclose(c1["poison_frac"], f)]
            if s.empty:
                continue
            k = int((s["accepted"] == want).sum())
            ok = k == len(s) if need == "all" else k >= 2
            add(h, f"f={f}: poisoned C1 at E1 {'accepted' if want else 'rejected'} "
                   f"({'every seed' if need == 'all' else '>= 2 of 3 seeds'})", k, len(s),
                "supported" if ok else "NOT supported")
    r1 = sw[(sw["attempt"] == "R1")]
    add("H3", "when C1 was rejected, soft R1 passed the gate (descriptive)", int(r1["accepted"].sum()),
        len(r1), "no R1 ran" if r1.empty else "descriptive")
    if not diag.empty:
        d = diag[(diag["arm"] == "poisoned") & (diag["method"] == "soft") & diag["run"].str.match(NAME)]
        ok = (d["w_flipped"] < 0.05) & (d["w_clean"] > 0.95)
        add("H4", "forced soft R1: mean weight < 0.05 on flipped rows and > 0.95 on clean T-known rows",
            int(ok.sum()), len(d), "supported" if len(d) and ok.all() else "NOT supported")
        f1 = diag[(diag["arm"] == "poisoned") & (diag["experience"] == 1) & diag["run"].str.match(NAME)
                  & np.isclose(diag["poison_frac"], 1.0)]
        for m in ("soft", "threeway+relabel"):
            s = f1[f1["method"] == m]
            if len(s):
                add("H5", f"f=1.0, forced {m} R1 would pass the gate (diagnostic, descriptive)",
                    int(s["would_pass_gate"].sum()), len(s), "descriptive")
    cl = sw[(sw["arm"] == "clean") & (sw["attempt"] == "C1")]
    add("H6", "clean C1 accepted at every update", int(cl["accepted"].sum()), len(cl),
        "supported" if len(cl) and cl["accepted"].all() else "NOT supported")
    return pd.DataFrame(rows)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_decisions(dec: pd.DataFrame, target: str, path: Path):
    s = dec[dec["run"].str.match(NAME) & (dec["arm"] == "poisoned") & (dec["experience"] == 1)
            & (dec["attempt"] == "C1")]
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    fs = sorted(s["poison_frac"].unique())
    pos = {f: i for i, f in enumerate(fs)}
    for acc, mk, lab in ((True, "o", "accepted"), (False, "X", "rejected")):
        q = s[s["accepted"] == acc]
        jit = (q["seed"] - 43) * 0.08
        ax.scatter(q["poison_frac"].map(pos) + jit, q[f"{target}_val_recall"], marker=mk, s=60,
                   color=ARM_COLOR["poisoned"], edgecolor="white", linewidth=1.5, label=f"C1 {lab}", zorder=3)
    fl = s.groupby("poison_frac")[f"{target}_floor"].mean()
    ax.hlines(fl.values, [pos[f] - 0.3 for f in fl.index], [pos[f] + 0.3 for f in fl.index], color=MUTED,
              linestyle=":", linewidth=1.5, label="gate floor (ref - 0.05)")
    ax.set_xticks(range(len(fs)), [f"f = {f}" for f in fs])
    ax.set_ylabel(f"{target} val recall", color=MUTED, fontsize=8)
    style(ax, f"Poisoned C1 at E1: {target} validation recall per seed")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario", default="cii")
    ap.add_argument("--runs", default="sweep_*", help="glob of run names under runs/<scenario>/")
    ap.add_argument("--repro", default="repro_s42")
    ap.add_argument("--main-dir", default=None, help="folder of the completed main run (for H0)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--decisions", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    out = default_work_dir() if a.out is None else Path(a.out)
    root = out / "runs" / a.scenario
    rep = out / "sweep"
    rep.mkdir(exist_ok=True)
    runs = runs_in(root, a.runs) + runs_in(root, a.repro)
    assert runs, f"no runs matching {a.runs} under {root}"

    if a.decisions:
        dec, diag = [], []
        for r in runs:
            d1, d2 = decision_rows(r)
            dec += d1
            diag += d2
        dec, diag = pd.DataFrame(dec), pd.DataFrame(diag)
        target = load_json(runs[0] / "config.json")["args"]["poison_target"]
        repro = None
        if a.main_dir and (root / a.repro).exists():
            m = pd.DataFrame(decision_rows(Path(a.main_dir))[0])
            r = dec[dec["run"] == a.repro]
            key = ["arm", "experience", "attempt", "accepted"]
            same = m[key].reset_index(drop=True).equals(r[key].reset_index(drop=True))
            col = f"{target}_val_recall"
            repro = {"same": same, "max_abs_diff_val_recall": float(np.abs(
                m[col].to_numpy() - r[col].to_numpy()).max()) if len(m) == len(r) else None}
            print(f"repro vs main: same decisions = {same}; max |diff| {target} val recall = "
                  f"{repro['max_abs_diff_val_recall']}")
        hyp = hypotheses(dec, diag, repro)
        dec.to_csv(rep / "decisions.csv", index=False)
        diag.to_csv(rep / "diagnostics.csv", index=False)
        hyp.to_csv(rep / "hypotheses.csv", index=False)
        plot_decisions(dec, target, rep / "decisions.png")
        show = ["run", "arm", "experience", "attempt", "accepted", f"{target}_val_recall", f"{target}_floor",
                f"{target}_val_mean_p", "benign_far"]
        print(dec[show].to_string(index=False))
        if not diag.empty:
            print("\nFORCED R1 - DIAGNOSTIC ONLY (never a decision):")
            print(diag[["run", "arm", "experience", "method", "would_pass_gate", f"{target}_val_recall",
                        "w_flipped", "w_clean", "n_relabelled", "n_relabelled_to_true"]].to_string(index=False))
        print("\nPre-registered hypotheses (validation only):")
        print(hyp.to_string(index=False))
        print(f"\nwritten to {rep}. Commit decisions.csv BEFORE running --test.")
        return

    if not (rep / "decisions.csv").exists():
        raise SystemExit("refuse: run --decisions first (and commit decisions.csv) before --test")
    fin = []
    for r in runs:
        cfg = load_json(r / "config.json")
        for arm in ARM_COLOR:
            if (r / arm / "final_test.json").exists():
                t = load_json(r / arm / "final_test.json")["test"]
                fin.append({"run": r.name, "poison_frac": cfg["args"]["poison_frac"], "seed": cfg.get("seed", 42),
                            "arm": arm, "benign_far": t["benign_far"], "macro_recall": t["macro_recall"],
                            "n_test_rows": t["n_rows"], "test_rows_sha256": t["rows_sha256"],
                            **{f"recall_{k}": v for k, v in t["recall"].items()}})
    fin = pd.DataFrame(fin)
    assert fin["test_rows_sha256"].nunique() == 1, "runs were tested on different rows"
    fin.to_csv(rep / "final_test.csv", index=False)
    sw = fin[fin["run"].str.match(NAME)]
    cols = [c for c in fin.columns if c.startswith("recall_")] + ["benign_far"]
    summ = sw.groupby(["poison_frac", "arm"])[cols].agg(["mean", "min", "max"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ = summ.reset_index()
    summ.insert(2, "n_seeds", sw.groupby(["poison_frac", "arm"]).size().to_numpy())
    summ.to_csv(rep / "test_summary.csv", index=False)
    target = load_json(runs[0] / "config.json")["args"]["poison_target"]
    if not summ.empty:
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
        for j, arm in enumerate(summ["arm"].unique()):
            s = summ[summ["arm"] == arm]
            c = f"recall_{target}"
            ax.errorbar(s["poison_frac"] + (j - 1) * 0.004, s[f"{c}_mean"],
                        # clip: the mean of identical values can exceed them by 1 ulp
                        yerr=np.clip([s[f"{c}_mean"] - s[f"{c}_min"], s[f"{c}_max"] - s[f"{c}_mean"]], 0, None),
                        color=ARM_COLOR[arm], marker="o", markersize=7, linewidth=2, capsize=3,
                        markeredgecolor="white", label=arm)
        ax.set_xlabel("poison fraction f (bars: min-max over seeds)", color=MUTED, fontsize=8)
        ax.set_ylabel(f"final test {target} recall", color=MUTED, fontsize=8)
        style(ax, f"Final-model test recall of '{target}' after E2")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(rep / "test_syn.png")
        plt.close(fig)
    print(summ.to_string(index=False))
    print(f"\nwritten to {rep}")


if __name__ == "__main__":
    main()
