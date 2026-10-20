"""Stage 5 - Read-only explanations of the implemented model and of one finished run.

Nothing here trains, writes files, or modifies checkpoints. Checkpoints are loaded on CPU
and used only under torch.no_grad() for inspection and inference. Used by
notebooks/03_understand_our_model.ipynb.

Two input folders (both are nids_out/ trees):
  results  : nids_results.zip contents   -> runs/<scenario>/<run>/..., scalers, manifests
  prepared : results_01.zip contents     -> clean_*.parquet (needed only for row-level parts)
"""
from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import LABEL_COL, load_json
from s1b_experiences import CLIP, apply_scaler, cat_codes, slog

ARMS = ("clean", "poisoned", "poisoned_repair")


# ---- locating inputs ---------------------------------------------------------
def find_dir(marker: str, env: str) -> Path | None:
    """First nids_out-like folder containing `marker` (env var, /kaggle/working, /kaggle/input)."""
    if os.environ.get(env):
        return Path(os.environ[env])
    cands = ["/kaggle/working/nids_out/" + marker] + sorted(glob.glob(f"/kaggle/input/**/{marker}", recursive=True))
    for c in cands:
        if Path(c).exists():
            return Path(c).parents[marker.count("/")]
    return None


def load_run(results: Path, scenario: str = "cii", run: str = "main") -> dict:
    r = results / "runs" / scenario / run
    out = {"dir": r, "config": load_json(r / "config.json"), "A0": load_json(r / "A0_metrics.json"),
           "manifest": load_json(r / "run_manifest.json"), "rounds": {}, "final": {}, "buffers": {}}
    for arm in ARMS:
        if (r / arm / "rounds.json").exists():
            out["rounds"][arm] = load_json(r / arm / "rounds.json")
            out["final"][arm] = load_json(r / arm / "final_test.json")
            out["buffers"][arm] = {t: pd.read_csv(r / arm / f"buffer_before_E{t}.csv")
                                   for t in range(1, len(out["config"]["schedule"]))}
    return out


# ---- the model, rebuilt from a checkpoint ---------------------------------------
def model_from_ckpt(path: Path, args: dict):
    """(model in eval mode on CPU, head classes). Shapes come from the checkpoint itself."""
    import torch
    from s2_model import TabTransformer
    ck = torch.load(path, map_location="cpu")
    st = ck["state"]
    n_num = st["tokenizer.W"].shape[0]
    card = [st[k].shape[0] for k in sorted(st) if k.startswith("tokenizer.cat_emb.")]
    m = TabTransformer(n_num, len(ck["classes"]), card, d=args["d"], heads=args["heads"],
                       layers=args["layers"], dropout=args["dropout"])
    m.load_state_dict(st)
    return m.eval(), list(ck["classes"])


def parameter_table(model) -> pd.DataFrame:
    """Every tensor trained by gradient descent, grouped by role."""
    def role(n):
        if n.startswith("tokenizer.W") or n.startswith("tokenizer.b"):
            return "numeric tokenizer (per-feature W_j, b_j)"
        if n.startswith("tokenizer.cat_emb"):
            return "protocol embedding table"
        if n.startswith("tokenizer.cls"):
            return "[CLS] token"
        if ".self_attn." in n:
            return "attention (W_Q, W_K, W_V packed + output projection)"
        if ".linear1." in n or ".linear2." in n:
            return "feed-forward (d -> 4d -> d)"
        if ".norm" in n or n.startswith("norm."):
            return "LayerNorm gain/bias"
        if n.startswith("head."):
            return "classifier head"
        return "other"
    rows = [{"name": n, "shape": tuple(p.shape), "n_params": p.numel(), "role": role(n)}
            for n, p in model.named_parameters()]
    return pd.DataFrame(rows)


def shape_trace(model, B: int = 4) -> pd.DataFrame:
    """Tensor shape after each stage, using random inputs (shapes only, no data)."""
    import torch
    n_num = model.tokenizer.W.shape[0]
    cards = [e.num_embeddings for e in model.tokenizer.cat_emb]
    x = torch.randn(B, n_num)
    c = torch.stack([torch.randint(0, k, (B,)) for k in cards], 1) if cards else torch.zeros(B, 0, dtype=torch.long)
    rows = [("x_num (scaled numeric)", tuple(x.shape)), ("x_cat (protocol code)", tuple(c.shape))]
    with torch.no_grad():
        t = model.tokenizer(x, c)
        rows.append(("tokens = [CLS | 77 numeric | protocol]", tuple(t.shape)))
        h = t
        for i, layer in enumerate(model.encoder.layers):
            h = layer(h)
            rows.append((f"after Transformer block {i + 1}", tuple(h.shape)))
        z = model.norm(h[:, 0])
        rows.append(("LayerNorm(h[:, 0])  (final [CLS])", tuple(z.shape)))
        rows.append(("logits = head(...)", tuple(model.head(z).shape)))
    return pd.DataFrame(rows, columns=["stage", "shape"])


# ---- preprocessing (statistics, not learned parameters) ----------------------
def scaler_table(sc: dict) -> pd.DataFrame:
    return pd.DataFrame({"feature": sc["features_numeric"], "mean_of_slog": sc["mean"], "std_of_slog": sc["std"],
                         "constant_in_E0": [f in sc["const_in_e0"] for f in sc["features_numeric"]]})


def worked_example(row: pd.DataFrame, sc: dict, features: list[str] | None = None) -> pd.DataFrame:
    """Every transform step for ONE row (raw -> slog -> standardized -> clip -> float32)."""
    feats = features or sc["features_numeric"]
    idx = [sc["features_numeric"].index(f) for f in feats]
    x = row[sc["features_numeric"]].to_numpy("float64")[0]
    s = slog(x)
    z = (s - np.array(sc["mean"])) / np.array(sc["std"])
    k = np.clip(z, -sc["clip"], sc["clip"])
    return pd.DataFrame({"feature": feats, "raw x": x[idx], "slog(x)": s[idx], "mean": np.array(sc["mean"])[idx],
                         "std": np.array(sc["std"])[idx], "z": z[idx], "clip(z, +-5)": k[idx],
                         "float32 input": k.astype("float32")[idx]})


def protocol_example(row: pd.DataFrame, sc: dict) -> dict:
    col = sc["features_categorical"][0]
    raw = row[col].iloc[0]
    return {"raw protocol": raw, "vocabulary (value -> id)": sc["cat_vocab"][col],
            "integer id fed to nn.Embedding": int(cat_codes(row, sc)[0, 0]),
            "unknown values map to": 0, "embedding table rows": len(sc["cat_vocab"][col]) + 1}


# ---- gate, recomputed from the saved metrics ---------------------------------
def gate_check(rounds: list[dict], args: dict) -> pd.DataFrame:
    """Recompute every gate inequality from the saved val metrics; must match the saved decision."""
    rows = []
    for r in rounds:
        v, ref = r["val"], r["ref_val"]
        checks = [(f"old {k}: r_C >= r_ref - {args['gate_old_drop']}", v["recall"].get(k, 0.0),
                   ref["recall"][k] - args["gate_old_drop"], ">=")
                  for k in r["old_classes"] if k != "BENIGN"]
        checks += [(f"new {k}: r_C >= {args['gate_new_min']}", v["recall"].get(k, 0.0), args["gate_new_min"], ">=")
                   for k in r["new_classes"]]
        checks += [(f"benign FAR <= {args['gate_far_max']}", v["benign_far"], args["gate_far_max"], "<="),
                   (f"new-class FAR <= {args['gate_newfar_max']}", v["new_class_far"], args["gate_newfar_max"], "<=")]
        if args.get("gate_conf_drop") is not None:  # only runs that switched the confidence check on
            checks += [(f"old {k}: mean p >= ref - {args['gate_conf_drop']}", v["mean_conf"].get(k, 0.0),
                        ref["mean_conf"][k] - args["gate_conf_drop"], ">=")
                       for k in r["old_classes"] if k != "BENIGN"]
        ok_all = True
        for name, val, bound, op in checks:
            ok = val >= bound if op == ">=" else val <= bound
            ok_all &= ok
            rows.append({"experience": f"E{r['experience']}", "attempt": r["attempt"], "check": name,
                         "value": val, "bound": bound, "margin": (val - bound) if op == ">=" else (bound - val),
                         "pass": ok})
        assert ok_all == r["accepted"], "recomputed gate disagrees with the saved decision"
    return pd.DataFrame(rows)


# ---- weighted loss arithmetic ------------------------------------------------
def class_weights(counts: dict) -> dict:
    """c_k = sqrt(N / (K * n_k)), exactly as s3_continual.train computes it."""
    n, k = sum(counts.values()), len(counts)
    return {c: math.sqrt(n / (k * v)) for c, v in counts.items()}


def training_label_counts(run: dict, prepared_counts: dict, arm: str, t: int) -> dict:
    """Stored-label counts the attempt at E_t trains on: current experience + replay buffer."""
    cur = dict(prepared_counts[f"E{t}"])
    for lab, n in run["buffers"][arm][t]["label"].value_counts().items():
        cur[lab] = cur.get(lab, 0) + int(n)
    return cur


# ---- row-level diagnostics (need the prepared parquet files) -------------------
def _rows(prepared: Path, split: str, rids, sc: dict, class_map: dict):
    d = pd.read_parquet(prepared / f"clean_{split}.parquet")
    d = d.set_index("_rid").loc[np.asarray(rids)].reset_index()
    x, c, _ = apply_scaler(d, sc, class_map)
    return x, c, d[LABEL_COL].to_numpy().astype(str)


def _proba(model, x, c):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(x), torch.from_numpy(c)), -1).numpy()


def buffer_probabilities(results: Path, prepared: Path, run: dict, scenario: str = "cii") -> pd.DataFrame:
    """Inference only, one row per (model, buffer row) of the poisoned arm's E1 buffer:
    probability of the stored label and of the true label, and the prediction. Models: T,
    and the accepted A1 of the clean and poisoned arms (A1 = what was trained on this buffer)."""
    sc = load_json(results / scenario / "scaler.json")
    cm = load_json(results / "class_map.json")
    buf = run["buffers"]["poisoned"][1]
    x, c, _ = _rows(prepared, "train", buf["rid"], sc, cm)
    args = run["config"]["args"]
    models = {"T (frozen teacher)": run["dir"] / "T.pt", "A1 clean": run["dir"] / "clean" / "A1.pt",
              "A1 poisoned": run["dir"] / "poisoned" / "A1.pt"}
    group = np.where(buf["flipped"], "flipped (stored BENIGN, true Syn)",
                     np.where(buf["label"] == "Syn", "clean Syn (stored Syn)", "clean BENIGN (stored BENIGN)"))
    out = []
    for name, path in models.items():
        if not path.exists():
            continue
        m, cls = model_from_ckpt(path, args)
        pr = _proba(m, x, c)
        pos = {k: i for i, k in enumerate(cls)}
        n = np.arange(len(buf))
        out.append(pd.DataFrame({
            "model": name, "group": group,
            "p_stored": pr[n, buf["label"].map(pos).to_numpy()],
            "p_true": pr[n, buf["true_label"].map(pos).to_numpy()],
            "pred": np.array(cls)[pr.argmax(1)], "stored": buf["label"].to_numpy(),
            "true": buf["true_label"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def poison_diagnostics(probs: pd.DataFrame) -> pd.DataFrame:
    """Summary of buffer_probabilities. For T, 'mean p(stored label)' is exactly the R1 weight
    the repair WOULD have given each group (R1 never ran in this run: diagnostic only)."""
    p = probs.assign(is_stored=probs["pred"] == probs["stored"], is_true=probs["pred"] == probs["true"])
    g = p.groupby(["model", "group"], sort=False)
    return pd.DataFrame({"n": g.size(), "mean p(stored label)": g["p_stored"].mean(),
                         "mean p(true label)": g["p_true"].mean(),
                         "share predicted = stored": g["is_stored"].mean(),
                         "share predicted = true": g["is_true"].mean()}).reset_index()


def contradiction_optimum(counts: dict, a: str = "Syn", b: str = "BENIGN") -> float:
    """If identical inputs carry label a in some rows and b in others, the weighted cross-entropy
    is minimised (for that input) at p(a) = c_a n_a / (c_a n_a + c_b n_b), with n = rows per label
    on that input. Here n_a = n_b (half the Syn exemplars flipped), so p(a) = c_a / (c_a + c_b).
    An idealisation: real flipped and clean rows are distinct inputs, only similar."""
    cw = class_weights(counts)
    return cw[a] / (cw[a] + cw[b])


def paired_test(results: Path, prepared: Path, run: dict, cls: str = "Syn", scenario: str = "cii",
                arms=("clean", "poisoned"), after: str | None = None) -> dict:
    """Exact McNemar test on the SAME test rows of one class: final model of arm a vs arm b."""
    from scipy.stats import binomtest
    sc = load_json(results / scenario / "scaler.json")
    cm = load_json(results / "class_map.json")
    excl = set(load_json(results / "protected_exclusions.json")["excluded_rids"])
    d = pd.read_parquet(prepared / "clean_test.parquet", columns=["_rid", LABEL_COL])
    rids = d.loc[(d[LABEL_COL] == cls) & ~d["_rid"].isin(excl), "_rid"].to_numpy()
    x, c, _ = _rows(prepared, "test", rids, sc, cm)
    after = after or f"A{len(run['config']['schedule']) - 1}"
    right = {}
    for arm in arms:
        m, k = model_from_ckpt(run["dir"] / arm / f"{after}.pt", run["config"]["args"])
        right[arm] = np.array(k)[_proba(m, x, c).argmax(1)] == cls
    a, b = arms
    only_a, only_b = int((right[a] & ~right[b]).sum()), int((~right[a] & right[b]).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"class": cls, "rows": int(len(rids)), f"recall {a}": float(right[a].mean()),
            f"recall {b}": float(right[b].mean()), f"right only in {a}": only_a,
            f"right only in {b}": only_b, "exact McNemar p": float(p)}


def example_row(prepared: Path, results: Path, scenario: str = "cii", label: str = "Syn"):
    """First E0 training row of `label` (by row id) with at least one clipped feature."""
    sc = load_json(results / scenario / "scaler.json")
    e0 = np.load(results / scenario / "experience_rids.npz")["e0"]
    d = pd.read_parquet(prepared / "clean_train.parquet")
    d = d[d["_rid"].isin(e0) & (d[LABEL_COL] == label)].sort_values("_rid")
    z = (slog(d[sc["features_numeric"]].to_numpy("float64")) - np.array(sc["mean"])) / np.array(sc["std"])
    i = int(np.flatnonzero((np.abs(z) > CLIP).any(axis=1))[0])
    return d.iloc[[i]]


# ---- plots (displayed only; nothing is saved) -----------------------------------
# Same validated arm colours as s4_report.py
ARM_COLOR = {"clean": "#2a78d6", "poisoned": "#eb6834", "poisoned_repair": "#1baf7a"}
ARM_STYLE = {"clean": "-", "poisoned": (0, (5, 3)), "poisoned_repair": (0, (1, 3))}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def deployed_val(run: dict) -> pd.DataFrame:
    """Val recall per class of the model in service after E0, E1, E2, per arm."""
    rows = []
    for arm, recs in run["rounds"].items():
        mets = {0: run["A0"]["val"]}
        for t in sorted({r["experience"] for r in recs}):
            rs = [r for r in recs if r["experience"] == t]
            acc = [r for r in rs if r["accepted"]]
            mets[t] = acc[0]["val"] if acc else rs[0]["ref_val"]
        for t, v in mets.items():
            rows += [{"arm": arm, "after": t, "class": k, "recall": rc} for k, rc in v["recall"].items()]
    return pd.DataFrame(rows)


def plot_recall(run: dict, ymin: float = 0.98):
    """Small multiples: one panel per class, val recall of the deployed model after each round."""
    import matplotlib.pyplot as plt
    dv = deployed_val(run)
    classes = ["BENIGN"] + [k for e in run["config"]["schedule"] for k in e]
    fig, axes = plt.subplots(1, len(classes), figsize=(3.0 * len(classes), 3.2), dpi=110, sharey=True)
    floor = None
    for ax, k in zip(axes, classes):
        for arm in run["rounds"]:
            s = dv[(dv["arm"] == arm) & (dv["class"] == k)].sort_values("after")
            ax.plot(s["after"], s["recall"], color=ARM_COLOR[arm], linestyle=ARM_STYLE[arm], linewidth=2,
                    marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=arm)
        if k == run["config"]["args"]["poison_target"]:
            ref = dv[(dv["arm"] == "clean") & (dv["class"] == k) & (dv["after"] == 0)]["recall"].iloc[0]
            floor = ref - run["config"]["args"]["gate_old_drop"]
            ax.annotate(f"gate floor at E1 = {floor:.3f}\n(far below this axis)", (0.05, 0.04),
                        xycoords="axes fraction", fontsize=7, color=MUTED)
        ts = range(len(run["config"]["schedule"]))
        ax.set_xticks(list(ts), [f"E{t}" for t in ts])
        ax.set_ylim(ymin, 1.0005)
        _style(ax, k)
    axes[0].set_ylabel(f"val recall (axis starts at {ymin})", color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Deployed model after each round (poisoned and poisoned_repair coincide: R1 never ran)",
                 x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig


def plot_confidence(probs: pd.DataFrame):
    """p(Syn | x) on the 2,000 true-Syn buffer rows, A1 clean vs A1 poisoned."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=110)
    bins = np.linspace(0, 1, 41)
    for name, arm in (("A1 clean", "clean"), ("A1 poisoned", "poisoned")):
        s = probs[(probs["model"] == name) & (probs["true"] == "Syn")]["p_true"]
        ax.hist(s, bins=bins, color=ARM_COLOR[arm], alpha=0.75, label=f"{name} (mean {s.mean():.3f})",
                edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("p > 0.5 guarantees\nargmax = Syn", (0.51, 0.95), xycoords=("data", "axes fraction"),
                fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("p(Syn | x) on true-Syn buffer rows (1,000 flipped + 1,000 clean)", color=MUTED, fontsize=8)
    ax.set_ylabel("rows", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _style(ax, "Poison lowered confidence, not the decision")
    fig.tight_layout()
    return fig
