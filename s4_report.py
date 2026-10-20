"""Stage 4 - Tables and plots from one Stage 3 run (no training, reads saved JSON only).

Writes into runs/<scenario>/<run>/report/:
  attempts.csv         one row per attempt (arm, E_t, C1/R1): gate decision + reasons, key metrics
  per_class_val.csv    val recall per class of the model deployed after each round, per arm
  final_test.csv       test metrics of each arm's final model on the SAME rows (BENIGN + all
                       scheduled classes; a class the arm never learned scores recall 0)
  target_recall.png    poison-target recall (val) of the deployed model, A0 -> E1 -> E2, per arm
  per_class_E<t>.png   val recall per class after E_t, grouped by arm

Usage:
  python s4_report.py --scenario cii --run main
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import default_work_dir, load_json

# Validated categorical slots (dataviz reference palette, light mode), fixed per arm.
ARM_COLOR = {"clean": "#2a78d6", "poisoned": "#eb6834", "poisoned_repair": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def deployed(records):
    """{t: val metrics of the model in service after round t} (accepted attempt, else ref)."""
    out = {}
    for t in sorted({r["experience"] for r in records}):
        rs = [r for r in records if r["experience"] == t]
        acc = [r for r in rs if r["accepted"]]
        out[t] = acc[0]["val"] if acc else rs[0]["ref_val"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario", choices=["ci", "cii"], default="cii")
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    out = default_work_dir() if a.out is None else Path(a.out)
    rdir = out / "runs" / a.scenario / a.run
    rep = rdir / "report"
    rep.mkdir(exist_ok=True)
    cfg = load_json(rdir / "config.json")
    target = cfg["args"]["poison_target"]
    a0 = load_json(rdir / "A0_metrics.json")["val"]
    arms = [p.name for p in rdir.iterdir() if (p / "rounds.json").exists()]
    arms = [k for k in ARM_COLOR if k in arms]

    att, pcv, fin, dep = [], [], [], {}
    for arm in arms:
        recs = load_json(rdir / arm / "rounds.json")
        for r in recs:
            v = r["val"]
            att.append({"arm": arm, "experience": f"E{r['experience']}", "attempt": r["attempt"],
                        "accepted": r["accepted"], "gate_reasons": "; ".join(r["gate_reasons"]),
                        f"{target}_recall": v["recall"].get(target), "benign_far": v["benign_far"],
                        "n_benign_val": v["n_benign"], "new_class_far": v["new_class_far"],
                        "macro_recall": v["macro_recall"],
                        "new_recall_min": min(v["recall"].get(k, 0) for k in r["new_classes"]),
                        "replay_flipped": r["n_replay_flipped"],
                        "w_flipped": (r["teacher_weights"] or {}).get("mean_w_flipped"),
                        "w_clean": (r["teacher_weights"] or {}).get("mean_w_clean_known")})
        dep[arm] = {0: a0, **deployed(recs)}
        for t, v in dep[arm].items():
            for k, rc in v["recall"].items():
                pcv.append({"arm": arm, "after": f"E{t}", "class": k, "val_recall": rc,
                            "support": v["support"].get(k)})
        fj = load_json(rdir / arm / "final_test.json")
        ft = fj["test"]
        fin.append({"arm": arm, "classes_learned": " ".join(fj["final_classes"]),
                    "accuracy": ft["accuracy"], "macro_recall": ft["macro_recall"],
                    "benign_far": ft["benign_far"], "n_benign_test": ft["n_benign"],
                    "n_test_rows": ft["n_rows"], "test_rows_sha256": ft.get("rows_sha256"),
                    **{f"recall_{k}": v for k, v in ft["recall"].items()}})
    assert len({f["test_rows_sha256"] for f in fin}) == 1, "arms were tested on different rows"
    pd.DataFrame(att).to_csv(rep / "attempts.csv", index=False)
    pcv = pd.DataFrame(pcv)
    pcv.to_csv(rep / "per_class_val.csv", index=False)
    pd.DataFrame(fin).to_csv(rep / "final_test.csv", index=False)

    # target-class recall across rounds: the headline poisoning effect
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ends = {}
    for j, arm in enumerate(arms):
        ts = sorted(dep[arm])
        ys = [dep[arm][t]["recall"].get(target, float("nan")) for t in ts]
        # later arms dashed so an identical line underneath stays visible
        ax.plot(ts, ys, color=ARM_COLOR[arm], linewidth=2, linestyle=["-", (0, (5, 3)), (0, (1, 3))][j % 3],
                marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2, label=arm)
        ends.setdefault(round(ys[-1], 2), []).append(arm)
    last = sorted(dep[arms[0]])[-1]
    placed = []
    for y in sorted(ends):  # one label per distinct end value, nudged apart
        ty = max([y] + [p + 0.07 for p in placed])
        placed.append(ty)
        ax.annotate(f"{' = '.join(ends[y])}: {y:.2f}", (last, y), xytext=(last + 0.08, ty),
                    textcoords="data", va="center", fontsize=8, color=INK)
    ts = sorted(dep[arms[0]])
    ax.set_xticks(ts, [f"after E{t}" for t in ts])
    ax.set_xlim(ts[0] - 0.2, ts[-1] + 0.9)
    style(ax, f"Validation recall of poison target '{target}' (deployed model)", "recall")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(rep / "target_recall.png")
    plt.close(fig)

    # per-class recall after each update, grouped by arm
    for t in sorted({int(s[1:]) for s in pcv["after"]} - {0}):
        sub = pcv[pcv["after"] == f"E{t}"]
        classes = ["BENIGN"] + [k for e in cfg["schedule"][:t + 1] for k in e]  # all scheduled so far
        fig, ax = plt.subplots(figsize=(max(6.4, 0.9 * len(classes) + 2), 3.8), dpi=150)
        bw = 0.8 / len(arms)
        for j, arm in enumerate(arms):
            s = sub[sub["arm"] == arm].set_index("class")["val_recall"]
            xs = [i + (j - (len(arms) - 1) / 2) * bw for i in range(len(classes))]
            ys = [s.get(k, 0.0) for k in classes]
            ax.bar(xs, ys, width=bw - 0.03, color=ARM_COLOR[arm], label=arm, edgecolor="white", linewidth=1)
            for x, k, y in zip(xs, classes, ys):
                if k not in s.index:  # model in service never learned this class
                    txt = "n/l"
                elif k == target or y == 0:  # class under attack, or a bar too short to see
                    txt = f"{y:.2f}"
                else:
                    continue
                ax.annotate(txt, (x, y), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=MUTED if txt == "n/l" else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        style(ax, f"Validation recall per class after E{t}", "recall")
        fig.text(0.99, 0.02, "n/l = class not learned by the model in service (update rejected)",
                 ha="right", fontsize=7, color=MUTED)
        ax.legend(frameon=False, fontsize=8, ncol=len(arms), loc="upper left", bbox_to_anchor=(0, -0.25))
        fig.tight_layout()
        fig.savefig(rep / f"per_class_E{t}.png")
        plt.close(fig)

    print(pd.DataFrame(att).drop(columns=["gate_reasons"]).to_string(index=False))
    print(f"\nreport written to {rep}")


if __name__ == "__main__":
    main()
