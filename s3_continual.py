"""Stage 3 - Continual updates: gate, replay-label poisoning, teacher-weighted repair.

Uses the Stage 1/1b outputs of ONE scenario (ci or cii). Classes per experience come from
that scenario's experience_manifest.json; nothing about the dataset is assumed here.

Initial round (E0): train a model on E0, save it twice as two separate files:
  T.pt  - frozen teacher. Never trained again; its weights are hashed before and after.
  A0.pt - first accepted model, the starting point for every arm.

Three arms share T and A0 and the same replay sampling seed:
  clean           : buffer B untouched; one attempt (C1) per update.
  poisoned        : before the first update, the attacker flips the stored label of a bounded
                    fraction of ONE old class in B (only B: D, val, test, T are out of reach).
  poisoned_repair : same poison; if the gate rejects C1, one repair attempt R1.

Per update t:
  C1 = copy of the last accepted model + new head rows, trained on D_t (new classes, plus
       fresh benign in CII) together with B.
  Gate on protected VAL rows (reference = last accepted model on the same rows):
       every old attack class keeps recall >= ref - gate_old_drop, every new class has
       recall >= gate_new_min, benign false-alarm rate <= gate_far_max, and the share of
       non-new rows predicted as a new class <= gate_newfar_max.
  R1 restarts from the last accepted model (A0 at the first update), never from C1. Replay rows
       whose stored label is a class T knows get weight p_T(stored label | x); rows of classes
       T never saw keep weight 1. T is only read.
  Accepted -> A_t, and only then B gains rows of E_t. Rejected -> keep the last accepted
       model, B unchanged.
Test rows are scored only at the end for the report, never for a decision: BENIGN + every
scheduled class, the same rows for every arm (a class an arm never learned scores recall 0).
Val/test never include Stage 1b's protected exclusions (rows whose model input equals a
training input); protected_exclusions.json must match the scenario's experience manifest.
The run records sha256 of every Stage 1/1b input; --resume refuses if any of them changed.

Head indices follow arrival order (BENIGN, E0 classes, E1 classes, ...) and are stored in
every checkpoint; class_map.json is still the data-side map.

Usage:
  python s3_continual.py --scenario cii --poison-target DNS
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import LABEL_COL, SEED, default_work_dir, load_json, save_json
from s1b_experiences import apply_scaler
from s2_model import TabTransformer

ARMS = {"clean": (False, False), "poisoned": (True, False), "poisoned_repair": (True, True)}


class Rows:
    """Scaled features of one split, addressable by row id (_rid)."""

    def __init__(self, df: pd.DataFrame, scaler: dict, class_map: dict):
        df = df.sort_values("_rid")
        self.x, self.c, _ = apply_scaler(df, scaler, class_map)
        self.lab = df[LABEL_COL].to_numpy().astype(str)
        self.rid = df["_rid"].to_numpy()

    def idx(self, rids) -> np.ndarray:
        rids = np.asarray(rids)
        i = np.searchsorted(self.rid, rids)
        assert len(rids) == 0 or (i.max() < len(self.rid) and np.array_equal(self.rid[i], rids)), \
            "row id not found in this split"
        return i


# ---- model helpers ---------------------------------------------------------
def build(n_out, ctx, a):
    return TabTransformer(ctx["n_num"], n_out, ctx["card"], d=a.d, heads=a.heads,
                          layers=a.layers, dropout=a.dropout).to(ctx["device"])


def save_ckpt(path, model, classes, meta=None):
    torch.save({"state": model.state_dict(), "classes": list(classes), "meta": meta or {}}, path)


def load_ckpt(path, ctx, a):
    ck = torch.load(path, map_location=ctx["device"])
    m = build(len(ck["classes"]), ctx, a)
    m.load_state_dict(ck["state"])
    return m, list(ck["classes"])


def state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rid_hash(rids) -> str:
    """Fingerprint of exactly which rows were scored."""
    return hashlib.sha256(np.sort(np.asarray(rids, dtype="int64")).tobytes()).hexdigest()


@torch.no_grad()
def predict_proba(model, x, c, device, bs=4096):
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(x[i:i + bs]).to(device),
                               torch.from_numpy(c[i:i + bs]).to(device)).float(), -1).cpu()
           for i in range(0, len(x), bs)]
    return torch.cat(out).numpy() if out else np.zeros((0, model.head.out_features), "float32")


def train(model, classes, x, c, lab, w, epochs, a, device, seed):
    """Weighted cross-entropy. Class weights (inverse sqrt frequency) offset scarce benign;
    w is the per-row weight (1 everywhere except teacher-weighted replay rows in R1)."""
    pos = {k: i for i, k in enumerate(classes)}
    y = pd.Series(lab).map(pos).to_numpy()
    assert not np.isnan(y.astype(float)).any(), "training label outside the model's classes"
    y = y.astype("int64")
    cnt = np.bincount(y, minlength=len(classes))
    cw = np.where(cnt > 0, np.sqrt(len(y) / (len(classes) * np.maximum(cnt, 1))), 0.0)
    sw = torch.tensor(cw[y] * w, dtype=torch.float32, device=device)
    X, C = torch.from_numpy(x).to(device), torch.from_numpy(c).to(device)
    Y = torch.from_numpy(y).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(device)
        tot, n = 0.0, 0
        for i in range(0, len(y), a.batch):
            b = perm[i:i + a.batch]
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(X[b], C[b])
            ce = F.cross_entropy(logits.float(), Y[b], reduction="none")
            loss = (ce * sw[b]).sum() / sw[b].sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        losses.append(tot / max(n, 1))
    return losses


def evaluate(model, classes, rows: Rows, device, new=(), scope=None):
    """Scores rows whose true class is in `scope` (default: the classes the model knows).
    A scope class the model lacks scores recall 0, so arms that rejected an update are
    still compared on the same rows. Benign false-alarm rate = share of benign rows predicted
    as any attack; new-class FAR = share of non-new rows predicted as a new class."""
    scope = list(classes) if scope is None else list(scope)
    m = np.isin(rows.lab, scope)
    true = rows.lab[m]
    pred = np.array(classes)[predict_proba(model, rows.x[m], rows.c[m], device).argmax(1)]
    recall = {k: float((pred[true == k] == k).mean()) for k in scope if (true == k).any()}
    support = {k: int((true == k).sum()) for k in scope}
    ben = true == "BENIGN"
    notnew = ~np.isin(true, list(new))
    conf = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="pred"))
    return {
        "n_rows": int(m.sum()), "rows_sha256": rid_hash(rows.rid[m]),
        "accuracy": float((pred == true).mean()),
        "macro_recall": float(np.mean(list(recall.values()))),
        "recall": recall, "support": support,
        "benign_far": float((pred[ben] != "BENIGN").mean()) if ben.any() else float("nan"),
        "n_benign": int(ben.sum()),
        "new_class_far": float(np.isin(pred[notnew], list(new)).mean()) if new and notnew.any() else 0.0,
        "confusion": {t: r.to_dict() for t, r in conf.iterrows()},
    }


def gate(met, ref, old_attack, new, a):
    reasons = []
    for k in old_attack:
        floor = ref["recall"][k] - a.gate_old_drop
        if met["recall"].get(k, 0.0) < floor:
            reasons.append(f"old {k} recall {met['recall'].get(k, 0):.3f} < floor {floor:.3f}")
    for k in new:
        if met["recall"].get(k, 0.0) < a.gate_new_min:
            reasons.append(f"new {k} recall {met['recall'].get(k, 0):.3f} < {a.gate_new_min}")
    if met["benign_far"] > a.gate_far_max:
        reasons.append(f"benign FAR {met['benign_far']:.3f} > {a.gate_far_max}")
    if met["new_class_far"] > a.gate_newfar_max:
        reasons.append(f"new-class FAR {met['new_class_far']:.3f} > {a.gate_newfar_max}")
    return not reasons, reasons


# ---- replay buffer and attack ---------------------------------------------
def buffer_add(buf, rows: Rows, rids, mem, rng):
    """Add up to `mem` rows per class of one accepted experience. Stores the TRUE label."""
    labs = rows.lab[rows.idx(rids)]
    parts = [buf]
    for k in sorted(set(labs)):
        sel = np.asarray(rids)[labs == k]
        if len(sel) > mem:
            sel = rng.choice(sel, size=mem, replace=False)
        parts.append(pd.DataFrame({"rid": np.sort(sel), "label": k, "true_label": k, "flipped": False}))
    buf = pd.concat(parts, ignore_index=True)
    buf["flipped"] = buf["flipped"].astype(bool)
    return buf


def poison(buf, target, to, frac, rng):
    """Flip the stored label of round(frac * n) buffer rows of ONE class. Bounded by frac."""
    assert 0 < frac <= 1, "poison fraction must be in (0, 1]"
    idx = np.flatnonzero((buf["label"] == target).to_numpy())
    assert len(idx), f"poison target {target} has no rows in the buffer"
    pick = rng.choice(idx, size=int(round(frac * len(idx))), replace=False)
    buf = buf.copy()
    buf.loc[pick, "label"] = to
    buf.loc[pick, "flipped"] = True
    return buf


def teacher_weights(T, T_classes, rows: Rows, buf, device):
    """p_T(stored label | x) for buffer rows whose stored label T knows; 1 otherwise."""
    i = rows.idx(buf["rid"].to_numpy())
    p = predict_proba(T, rows.x[i], rows.c[i], device)
    known = buf["label"].isin(T_classes).to_numpy()
    col = buf["label"][known].map({k: j for j, k in enumerate(T_classes)}).to_numpy().astype(int)
    w = np.ones(len(buf))
    w[known] = p[np.flatnonzero(known), col]
    fl = buf["flipped"].to_numpy().astype(bool)
    # flipped is used ONLY for this diagnostic, never for training
    stats = {"mean_w_flipped": float(w[fl].mean()) if fl.any() else None,
             "mean_w_clean_known": float(w[known & ~fl].mean()) if (known & ~fl).any() else None,
             "n_weighted": int(known.sum()), "n_unweighted": int((~known).sum())}
    return w, stats


# ---- experiment ------------------------------------------------------------
def initial_round(ctx, a, rdir):
    T_path, A0_path = rdir / "T.pt", rdir / "A0.pt"
    classes = ["BENIGN"] + ctx["schedule"][0]
    if a.resume and T_path.exists() and A0_path.exists():
        print("resume: loading T and A0")
    else:
        torch.manual_seed(SEED)
        m = build(len(classes), ctx, a)
        tr = ctx["train"]
        i = tr.idx(ctx["exp"][0])
        t0 = time.time()
        losses = train(m, classes, tr.x[i], tr.c[i], tr.lab[i], np.ones(len(i)), a.epochs0,
                       a, ctx["device"], SEED)
        met = evaluate(m, classes, ctx["val"], ctx["device"])
        save_ckpt(T_path, m, classes, {"role": "frozen teacher", "trained_on": "E0"})
        save_ckpt(A0_path, m, classes, {"role": "accepted A0", "trained_on": "E0"})
        save_json({"losses": losses, "seconds": time.time() - t0, "val": met}, rdir / "A0_metrics.json")
        print(f"E0 trained in {time.time() - t0:.0f}s | val recall {met['recall']} | "
              f"benign FAR {met['benign_far']:.3f}")
    T, T_classes = load_ckpt(T_path, ctx, a)
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)
    return T, T_classes


def run_arm(arm, ctx, a, rdir, T, T_classes):
    do_poison, do_repair = ARMS[arm]
    adir = rdir / arm
    adir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)  # same buffer draws in every arm
    A, A_cls = load_ckpt(rdir / "A0.pt", ctx, a)
    tr, val, dev = ctx["train"], ctx["val"], ctx["device"]
    buf = buffer_add(pd.DataFrame(columns=["rid", "label", "true_label", "flipped"]),
                     tr, ctx["exp"][0], a.mem_per_class, rng)
    records = []
    for t in range(1, len(ctx["schedule"])):
        if do_poison and t == 1:
            buf = poison(buf, a.poison_target, a.poison_to, a.poison_frac,
                         np.random.default_rng(SEED + 2))
        buf.to_csv(adir / f"buffer_before_E{t}.csv", index=False)
        new = ctx["schedule"][t]
        ref = evaluate(A, A_cls, val, dev)
        cand = A_cls + new
        di = tr.idx(ctx["exp"][t])
        bi = tr.idx(buf["rid"].to_numpy())
        x = np.concatenate([tr.x[di], tr.x[bi]])
        c = np.concatenate([tr.c[di], tr.c[bi]])
        lab = np.concatenate([tr.lab[di], buf["label"].to_numpy().astype(str)])
        accepted = False
        for att in ["C1"] + (["R1"] if do_repair else []):
            torch.manual_seed(SEED + 10 * t)
            m = copy.deepcopy(A)  # C1 and R1 both start from the last accepted model
            m.expand_head(len(cand))
            w = np.ones(len(lab))
            wstats = None
            if att == "R1":
                wb, wstats = teacher_weights(T, T_classes, tr, buf, dev)
                w[len(di):] = wb
            t0 = time.time()
            losses = train(m, cand, x, c, lab, w, a.epochs_update, a, dev, SEED + t)
            met = evaluate(m, cand, val, dev, new)
            ok, reasons = gate(met, ref, [k for k in A_cls if k != "BENIGN"], new, a)
            save_ckpt(adir / f"E{t}_{att}.pt", m, cand, {"arm": arm, "attempt": att, "accepted": ok})
            records.append({"arm": arm, "experience": t, "attempt": att, "accepted": ok,
                            "gate_reasons": reasons, "new_classes": new, "old_classes": A_cls,
                            "losses": losses, "seconds": time.time() - t0, "teacher_weights": wstats,
                            "n_current": int(len(di)), "n_replay": int(len(bi)),
                            "n_replay_flipped": int(buf["flipped"].sum()),
                            "val": met, "ref_val": ref})
            print(f"[{arm}] E{t} {att}: {'ACCEPT' if ok else 'REJECT'} {reasons}")
            if ok:
                A, A_cls, accepted = m, cand, True
                break
        if accepted:  # buffer grows only after acceptance
            buf = buffer_add(buf, tr, ctx["exp"][t], a.mem_per_class, rng)
        else:
            print(f"[{arm}] E{t}: no candidate accepted; keeping previous model, buffer unchanged")
        save_ckpt(adir / f"A{t}.pt", A, A_cls, {"arm": arm, "after": f"E{t}", "updated": accepted})
        save_json(records, adir / "rounds.json")
    test = evaluate(A, A_cls, ctx["test"], dev, scope=ctx["scope"])  # same rows in every arm
    save_json({"arm": arm, "final_classes": A_cls, "scope": ctx["scope"], "test": test},
              adir / "final_test.json")
    return records, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario", choices=["ci", "cii"], default="cii")
    ap.add_argument("--run", default="main", help="run name under runs/<scenario>/")
    ap.add_argument("--arms", default="clean,poisoned,poisoned_repair")
    ap.add_argument("--epochs0", type=int, default=10)
    ap.add_argument("--epochs-update", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mem-per-class", type=int, default=2000, help="buffer rows per class per experience")
    ap.add_argument("--poison-target", default=None, help="ONE old attack class (default: first E0 attack)")
    ap.add_argument("--poison-to", default="BENIGN")
    ap.add_argument("--poison-frac", type=float, default=0.5)
    ap.add_argument("--gate-old-drop", type=float, default=0.05)
    ap.add_argument("--gate-new-min", type=float, default=0.80)
    ap.add_argument("--gate-far-max", type=float, default=0.05)
    ap.add_argument("--gate-newfar-max", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true", help="reuse T/A0 if saved; skip finished arms")
    ap.add_argument("--dry-run", action="store_true", help="load and verify all inputs, then stop (no training)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out = default_work_dir() if a.out is None else Path(a.out)
    sdir = out / a.scenario
    em = load_json(sdir / "experience_manifest.json")
    schedule = em["schedule"]
    a.poison_target = a.poison_target or schedule[0][0]
    assert a.poison_target in schedule[0], "poison target must be an E0 attack class (T knows it)"
    assert a.poison_to in ["BENIGN"] + schedule[0], "poison_to must be a class known at E0"
    assert a.poison_to != a.poison_target

    # every arm and every resumed process must use these exact Stage 1/1b files
    inputs = {n: file_hash(out / n) for n in
              ["split_manifest.json", "class_map.json", "clean_train.parquet", "clean_val.parquet",
               "clean_test.parquet", "protected_exclusions.json", f"{a.scenario}/experience_rids.npz",
               f"{a.scenario}/scaler.json", f"{a.scenario}/experience_manifest.json"]}
    rdir = out / "runs" / a.scenario / a.run
    run_args = {k: v for k, v in vars(a).items() if k not in ("resume", "arms", "device", "dry_run")}
    if a.resume and (rdir / "config.json").exists():
        old = load_json(rdir / "config.json")
        changed = [n for n in inputs if old.get("inputs", {}).get(n) != inputs[n]]
        if changed or old["args"] != run_args:
            raise SystemExit(f"refuse --resume of '{a.run}': Stage 1/1b files changed {changed} or "
                             f"settings changed. Use a new --run name.")

    torch.manual_seed(SEED)
    scaler = load_json(sdir / "scaler.json")
    class_map = load_json(out / "class_map.json")
    z = np.load(sdir / "experience_rids.npz")
    exp = [z[f"e{t}"] for t in range(len(schedule))]
    used = np.unique(np.concatenate(exp))
    scope = ["BENIGN"] + [k for e in schedule for k in e]  # unscheduled classes never scored
    tr_df = pd.read_parquet(out / "clean_train.parquet")
    # protected rows whose model input equals a training input (Stage 1b); same ids for CI and CII
    pe = load_json(out / "protected_exclusions.json")
    excluded = np.array(pe["excluded_rids"], dtype="int64")
    assert hashlib.sha256(excluded.tobytes()).hexdigest() == pe["rids_sha256"] == \
        em["protected_exclusions"]["rids_sha256"], "protected_exclusions.json does not match this scenario's Stage 1b"

    def protected(sp):
        d = pd.read_parquet(out / f"clean_{sp}.parquet")
        return Rows(d[d[LABEL_COL].isin(scope) & ~d["_rid"].isin(excluded)], scaler, class_map)

    ctx = {
        "device": torch.device(a.device), "schedule": schedule, "exp": exp, "scope": scope,
        "train": Rows(tr_df[tr_df["_rid"].isin(used)], scaler, class_map),
        "val": protected("val"), "test": protected("test"),
        "n_num": len(scaler["features_numeric"]),
        "card": [len(v) + 1 for v in scaler["cat_vocab"].values()],
    }
    del tr_df
    assert not (set(used) & set(ctx["val"].rid) | set(used) & set(ctx["test"].rid)), \
        "training row id also in val/test"
    # the rows loaded here must be exactly the rows Stage 1b reported
    rows = {sp: {k: int((ctx[sp].lab == k).sum()) for k in scope} for sp in ("val", "test")}
    for sp in rows:
        assert rows[sp] == em["protected_exclusions"]["kept_by_split_class"][sp], \
            f"{sp} rows differ from Stage 1b: {rows[sp]}"
    per_exp = {f"E{t}": pd.Series(ctx["train"].lab[ctx["train"].idx(e)]).value_counts().to_dict()
               for t, e in enumerate(exp)}
    assert per_exp == em["counts_per_experience"], f"experience rows differ from Stage 1b: {per_exp}"
    print(f"scenario={a.scenario} schedule={schedule} device={ctx['device']} "
          f"features={ctx['n_num']} val_rows={len(ctx['val'].rid)} test_rows={len(ctx['test'].rid)}")
    print("train rows per experience:", per_exp)
    print("val rows:", rows["val"], "\ntest rows:", rows["test"])
    print(f"protected rows excluded: {len(excluded)} | inputs match Stage 1b: OK")
    if a.dry_run:
        print("dry run: everything loads and matches Stage 1b; nothing trained or written")
        return
    rdir.mkdir(parents=True, exist_ok=True)
    save_json({"args": run_args, "seed": SEED, "schedule": schedule, "inputs": inputs,
               "protected_excluded_rows": int(len(excluded)), "val_rows": rows["val"],
               "test_rows": rows["test"], "device": str(ctx["device"]), "torch": torch.__version__},
              rdir / "config.json")

    T, T_classes = initial_round(ctx, a, rdir)
    t_hash = state_hash(T)
    for arm in a.arms.split(","):
        if a.resume and (rdir / arm / "final_test.json").exists():
            print(f"resume: {arm} already finished")
            continue
        run_arm(arm, ctx, a, rdir, T, T_classes)
    t_after = state_hash(T)
    t_file = state_hash(load_ckpt(rdir / "T.pt", ctx, a)[0])
    assert t_hash == t_after == t_file, "teacher T changed!"
    A0, A0_cls = load_ckpt(rdir / "A0.pt", ctx, a)
    save_json({"T_sha256": t_hash, "T_unchanged": True, "T_classes": T_classes,
               "inputs": inputs,
               "A0_test": evaluate(A0, A0_cls, ctx["test"], ctx["device"], scope=ctx["scope"])},
              rdir / "run_manifest.json")
    print(f"done. teacher unchanged (sha256 {t_hash[:12]}). outputs in {rdir}")


if __name__ == "__main__":
    main()
