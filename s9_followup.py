"""Follow-up FU1 - can the clean blocks that the 12-label runs rejected be learned? (a separate experiment)

The all12 runs (notebook 04, part B) rejected CLEAN updates: E1 (new DNS recall < 0.80, DNS val rows predicted
LDAP) and E5 (old SSDP recall ~0, SSDP val rows predicted UDP). s3 then skips such a block for good. FU1 asks,
deciding on VALIDATION only:
  separability  1-nearest-neighbour recall for DNS/LDAP and SSDP/UDP: a ceiling hint, decides nothing
  probe         one block of the clean arm, from that arm's saved start model and buffer, one factor at a time
                (E1: update-training / class-balance changes; E5: balanced class weights or distillation)
  decide        the pre-registered selection rule -> decisions.json (one recipe per failure type, or none)
  policy        clean arm, whole schedule, with the retry policy below (validation decisions only, no test)
  score-test    the deployed model on test, ONCE, after the validation files are committed (needs that sha)

Inputs are read only: --src is a nids_out with clean_*.parquet, all12/<scenario>/ and
runs/<scenario>/all12_<scenario>_s42/ (its config.json must hash the same Stage 1/1b files). Outputs go only to
<work>/followup/<scenario>/. The frozen E0 teacher T is never loaded; its file hash is checked unchanged.

Recipes (fixed before any FU1 run; each changes ONE factor of s3's C1 update):
  base      s3's C1: 5 update epochs, lr 1e-3, class weight (N / (K n_k)) ** 0.5
  ep10      10 update epochs              ep20      20 update epochs          lr3e-4    learning rate 3e-4
  balanced  class weight power 1.0: every class, old (replayed) or new, gets the same total loss weight
  distill   + 1.0 * 2^2 * KL(last accepted model || candidate) on the old classes' logits (temperature 2)

Retry policy (`policy`; replaces "skip the block for good" in this experiment only):
  C1  every block first gets the base update, exactly as in s3.
  C2  if C1 is rejected: one retry of the same block with the recipe selected for its failure type. Old-class
      failure (an old class below its floor, benign FAR or new-class FAR) takes priority over new-class failure
      (a new class below gate_new_min). No recipe selected for that type -> no C2.
  D1  a block still rejected is retried once more (recipe for its last failure type, else base) from the latest
      accepted model and buffer, right after the next accepted block, or at the end of the schedule.
  After that the block is final: its classes are reported as not learned. The gate and its thresholds are
  unchanged; the buffer gains a block's rows only when that block is accepted. At most 3 attempts per block.

Usage (see notebooks/05_followup_fu1.ipynb):
  python s9_followup.py probe --src S --scenario cii --block 1 --recipes base,ep10,ep20,lr3e-4,balanced
"""
from __future__ import annotations

import argparse
import copy
import re
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import s3_continual as S3
from common import default_work_dir, load_json, save_json

RECIPES = {
    "base": {},
    "ep10": {"epochs_update": 10},
    "ep20": {"epochs_update": 20},
    "lr3e-4": {"lr": 3e-4},
    "balanced": {"cw_power": 1.0},
    "distill": {"kd_lambda": 1.0, "kd_temp": 2.0},
}
FAMILY = {"new_class": ["base", "ep10", "ep20", "lr3e-4", "balanced"],
          "old_class": ["base", "balanced", "distill"]}
RULE = ("per failure type, over all (scenario, seed) cells of its probe block: a non-base recipe is eligible if it "
        "passes the gate in >= ceil(2/3 of its cells); pick the eligible recipe with most passes, then the highest "
        "median gate margin, then the earlier one in the fixed recipe list; none eligible -> no recipe")


def setup(a):
    src = Path(a.src)
    run = src / "runs" / a.scenario / (a.source_run or f"all12_{a.scenario}_s42")
    cfg = load_json(run / "config.json")
    hp = Namespace(**cfg["args"])
    bad = [n for n in S3.input_names(a.tag, a.scenario) if S3.file_hash(src / n) != cfg["inputs"][n]]
    assert not bad, f"Stage 1/1b files differ from the ones the source run used: {bad}"
    dest = Path(a.work or default_work_dir()) / "followup" / a.scenario
    assert not dest.resolve().is_relative_to((src / "runs").resolve()), "never write into the s3 runs"
    dest.mkdir(parents=True, exist_ok=True)
    ctx = S3.load_context(src, a.tag, a.scenario, a.device)[0]
    return run, cfg, hp, ctx, dest


def failure_type(reasons: list[str]) -> str:
    return "old_class" if any(r.startswith(("old ", "benign FAR", "new-class FAR")) for r in reasons) else "new_class"


def candidate(ctx, hp, start, A_cls, t, rids, buf, recipe, seed):
    """Train one candidate for block t from `start` with one recipe and gate it on validation (as s3's attempt)."""
    r, tr, dev = RECIPES[recipe], ctx["train"], ctx["device"]
    new = ctx["schedule"][t]
    di, bi = tr.idx(rids), tr.idx(buf["rid"].to_numpy())
    x, c = np.concatenate([tr.x[di], tr.x[bi]]), np.concatenate([tr.c[di], tr.c[bi]])
    lab = np.concatenate([tr.lab[di], buf["label"].to_numpy().astype(str)])
    cand = A_cls + new
    hp_r = Namespace(**{**vars(hp), **{k: v for k, v in r.items() if k in ("epochs_update", "lr")}})
    ref = S3.evaluate(start, A_cls, ctx["val"], dev)
    distill = (copy.deepcopy(start), len(A_cls), r["kd_lambda"], r["kd_temp"]) if "kd_lambda" in r else None
    torch.manual_seed(seed + 10 * t)  # same seeds as s3's C1, so recipes are compared on paired seeds
    m = copy.deepcopy(start)
    m.expand_head(len(cand))
    t0 = time.time()
    losses = S3.train(m, cand, x, c, lab, np.ones(len(lab)), hp_r.epochs_update, hp_r, dev, seed + t,
                      cw_power=r.get("cw_power", 0.5), distill=distill)
    met = S3.evaluate(m, cand, ctx["val"], dev, new)
    old_attack = [k for k in A_cls if k != "BENIGN"]
    ok, reasons = S3.gate(met, ref, old_attack, new, hp)
    margins = {f"old {k}": met["recall"].get(k, 0.0) - (ref["recall"][k] - hp.gate_old_drop) for k in old_attack}
    margins |= {f"new {k}": met["recall"].get(k, 0.0) - hp.gate_new_min for k in new}
    margins |= {"benign FAR": hp.gate_far_max - met["benign_far"],
                "new-class FAR": hp.gate_newfar_max - met["new_class_far"]}
    binding = min(margins, key=margins.get)
    return m, cand, ok, {"recipe": recipe, "seed": seed, "block": t, "new_classes": new, "old_classes": list(A_cls),
                         "gate_pass": ok, "gate_reasons": reasons, "gate_margin": margins[binding],
                         "binding": binding, "seconds": time.time() - t0, "losses": losses,
                         "n_current": int(len(di)), "n_replay": int(len(bi)), "val": met, "ref_val": ref}


def flat(rec: dict) -> dict:
    v = rec["val"]
    return {"recipe": rec["recipe"], "seed": rec["seed"], "block": rec["block"], "gate_pass": rec["gate_pass"],
            "gate_margin": rec["gate_margin"], "binding": rec["binding"],
            "gate_reasons": "; ".join(rec["gate_reasons"]) or "-", "benign_far": v["benign_far"],
            "new_class_far": v["new_class_far"], "seconds": rec["seconds"],
            **{f"val_recall_{k}": v["recall"].get(k, 0.0) for k in rec["old_classes"] + rec["new_classes"]},
            **({"reproduces_logged_C1": rec["reproduces_logged_C1"]} if "reproduces_logged_C1" in rec else {})}


def probe(a):
    run, cfg, hp, ctx, dest = setup(a)
    t, seed0 = a.block, cfg.get("seed", 42)
    start, A_cls = S3.load_ckpt(run / "A0.pt" if t == 1 else run / "clean" / f"A{t - 1}.pt", ctx, hp)
    buf = pd.read_csv(run / "clean" / f"buffer_before_E{t}.csv")
    logged = next(r for r in load_json(run / "clean" / "rounds.json") if r["experience"] == t and r["attempt"] == "C1")
    assert logged["old_classes"] == A_cls and logged["new_classes"] == ctx["schedule"][t], "wrong start model"
    pdir = dest / f"probe_E{t}"
    pdir.mkdir(exist_ok=True)
    rows = []
    for recipe in a.recipes.split(","):
        assert recipe in RECIPES, f"unknown recipe {recipe}"
        for seed in map(int, a.seeds.split(",")):
            f = pdir / f"{recipe}_s{seed}.json"
            if f.exists():  # resumable: a finished probe is never retrained
                rec = load_json(f)
            else:
                rec = {"scenario": a.scenario, **candidate(ctx, hp, start, A_cls, t, ctx["exp"][t], buf, recipe, seed)[3]}
                if recipe == "base" and seed == seed0:  # must equal the logged s3 C1 of the source run
                    rec["reproduces_logged_C1"] = max(abs(rec["val"]["recall"][k] - logged["val"]["recall"][k])
                                                      for k in logged["val"]["recall"])
                save_json(rec, f)
            rows.append({"scenario": a.scenario, **flat(rec)})
            print(f"[{a.scenario}] E{t} {recipe:9s} s{seed}: {'PASS' if rec['gate_pass'] else 'fail'} "
                  f"margin {rec['gate_margin']:+.3f} ({rec['binding']}) {rec['seconds']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(dest / f"probe_E{t}.csv", index=False)
    print(f"written {dest / f'probe_E{t}.csv'} (validation only)")


def decide(a):
    root = Path(a.work or default_work_dir()) / "followup"
    out = {"rule": RULE}
    for fam, block in (("new_class", a.new_class_block), ("old_class", a.old_class_block)):
        files = sorted(root.glob(f"*/probe_E{block}.csv"))
        assert files, f"no probe_E{block}.csv under {root}"
        d = pd.concat([pd.read_csv(p) for p in files])
        tab = d.groupby("recipe").agg(cells=("gate_pass", "size"), passed=("gate_pass", "sum"),
                                      median_margin=("gate_margin", "median"), min_margin=("gate_margin", "min"))
        tab = tab.reindex([r for r in FAMILY[fam] if r in tab.index])
        tab["eligible"] = (tab.index != "base") & (tab["passed"] >= np.ceil(2 * tab["cells"] / 3))
        elig = tab[tab["eligible"]].sort_values(["passed", "median_margin"], ascending=False, kind="stable")
        out[fam] = {"block": block, "scenarios": [p.parent.name for p in files],
                    "recipe": elig.index[0] if len(elig) else None,
                    "table": tab.reset_index().to_dict("records")}
        print(f"{fam} (probe E{block}):\n{tab.to_string()}\n -> recipe: {out[fam]['recipe']}\n")
    save_json(out, root / "decisions.json")
    print(f"written {root / 'decisions.json'} (validation only). Commit it before `policy` results are tested.")


def policy(a):
    run, cfg, hp, ctx, dest = setup(a)
    dec = load_json(Path(a.decisions))
    recipe_for = {fam: dec[fam]["recipe"] for fam in FAMILY}
    rdir = dest / a.run
    if (rdir / "summary.json").exists():
        raise SystemExit(f"refuse: {rdir} is complete; it is never overwritten. Use a new --run name.")
    rdir.mkdir(parents=True, exist_ok=True)
    seed, t_hash = cfg.get("seed", 42), S3.file_hash(run / "T.pt")
    A, A_cls = S3.load_ckpt(run / "A0.pt", ctx, hp)
    rng = np.random.default_rng(seed + 1)  # same buffer draws as s3 until the schedules diverge
    buf = S3.buffer_add(pd.DataFrame(columns=["rid", "label", "true_label", "flipped"]), ctx["train"],
                        ctx["exp"][0], hp.mem_per_class, rng)
    records, pending, status = [], [], {}

    def attempt(t, att, recipe):
        nonlocal A, A_cls, buf
        m, cand, ok, rec = candidate(ctx, hp, A, A_cls, t, ctx["exp"][t], buf, recipe, seed)
        S3.save_ckpt(rdir / f"E{t}_{att}.pt", m, cand, {"block": t, "attempt": att, "recipe": recipe, "accepted": ok})
        records.append({**rec, "attempt": att, "accepted": ok})
        save_json(records, rdir / "rounds.json")
        print(f"[{a.scenario}] E{t} {att} ({recipe}): {'ACCEPT' if ok else 'REJECT'} {rec['gate_reasons']}", flush=True)
        if ok:
            A, A_cls = m, cand
            buf = S3.buffer_add(buf, ctx["train"], ctx["exp"][t], hp.mem_per_class, rng)
            status[t] = f"accepted at {att} ({recipe})"
        return ok, rec["gate_reasons"]

    def retry_pending():
        while pending:
            p, rcp = pending.pop(0)
            ok, _ = attempt(p, "D1", rcp)
            if not ok:
                status[p] = "rejected (C1, C2 if any, D1): not learned"

    for t in range(1, len(ctx["schedule"])):
        ok, reasons = attempt(t, "C1", "base")
        if not ok and recipe_for[failure_type(reasons)]:
            ok, reasons = attempt(t, "C2", recipe_for[failure_type(reasons)])
        if ok:
            retry_pending()
        else:
            pending.append((t, recipe_for[failure_type(reasons)] or "base"))
    retry_pending()
    assert S3.file_hash(run / "T.pt") == t_hash, "teacher file changed"
    S3.save_ckpt(rdir / "final.pt", A, A_cls, {"role": "deployed after the retry policy"})
    save_json({"source_run": str(run), "seed": seed, "recipes": recipe_for, "decisions_sha256": S3.file_hash(a.decisions),
               "inputs": cfg["inputs"], "T_sha256_file": t_hash, "T_unchanged": True, "final_classes": A_cls,
               "block_status": {f"E{t}": status.get(t) for t in range(1, len(ctx["schedule"]))},
               "device": str(ctx["device"]), "torch": torch.__version__}, rdir / "summary.json")
    print(f"deployed classes: {A_cls}\nwritten {rdir} (validation decisions only; no test scored)")


def score_test(a):
    assert re.fullmatch(r"[0-9a-f]{7,40}", a.decisions_commit or ""), \
        "--decisions-commit <git sha> of the commit that holds this run's validation files is required"
    run, cfg, hp, ctx, dest = setup(a)
    rdir = dest / a.run
    assert (rdir / "summary.json").exists(), "run `policy` first"
    if (rdir / "final_test.json").exists():
        raise SystemExit(f"refuse: {rdir} was already scored on test; it is scored once")
    m, cls = S3.load_ckpt(rdir / "final.pt", ctx, hp)
    test = S3.evaluate(m, cls, ctx["test"], ctx["device"], scope=ctx["scope"])
    save_json({"final_classes": cls, "decisions_commit": a.decisions_commit,
               "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test": test},
              rdir / "final_test.json")
    old = load_json(run / "clean" / "final_test.json")["test"]
    assert old["rows_sha256"] == test["rows_sha256"], "not the source run's test rows"
    print(pd.DataFrame({"s3 clean (skip rejected)": old["recall"], "FU1 retry policy": test["recall"]}).round(4)
          .to_string())
    print(f"benign FAR {old['benign_far']:.4f} -> {test['benign_far']:.4f}")


def separability(a):
    """Descriptive only: 1-NN recall of each class of a pair, training rows of this scenario vs its val rows."""
    run, cfg, hp, ctx, dest = setup(a)
    tr, va, dev = ctx["train"], ctx["val"], ctx["device"]
    ti = tr.idx(np.concatenate(ctx["exp"]))
    rows = []
    for pair in a.pairs.split(","):
        ks = pair.split(":")
        s, v = ti[np.isin(tr.lab[ti], ks)], np.flatnonzero(np.isin(va.lab, ks))
        # protocol id as a far-apart extra column, so a neighbour always shares the protocol
        X = torch.from_numpy(np.c_[tr.x[s], 100.0 * tr.c[s]]).float().to(dev)
        nn, dist = [], []
        for i in range(0, len(v), 2048):
            q = torch.from_numpy(np.c_[va.x[v[i:i + 2048]], 100.0 * va.c[v[i:i + 2048]]]).float().to(dev)
            d, j = torch.cdist(q, X).min(1)
            nn.append(j.cpu().numpy())
            dist.append(d.cpu().numpy())
        pred, dist, true = tr.lab[s][np.concatenate(nn)], np.concatenate(dist), va.lab[v]
        for k in ks:
            rows.append({"scenario": a.scenario, "pair": pair, "class": k, "val_rows": int((true == k).sum()),
                         "train_rows": int((tr.lab[s] == k).sum()), "nn_recall": float((pred[true == k] == k).mean()),
                         "share_identical_input": float((dist[true == k] == 0).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(dest / "separability.csv", index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["separability", "probe", "decide", "policy", "score-test"])
    ap.add_argument("--src", help="nids_out with clean_*.parquet, <tag>/<scenario>/ and the source run (read only)")
    ap.add_argument("--work", default=None, help="writes go to <work>/followup/ (default: the nids_out work dir)")
    ap.add_argument("--scenario", choices=["ci", "cii"], default="cii")
    ap.add_argument("--tag", default="all12")
    ap.add_argument("--source-run", default=None, help="default all12_<scenario>_s42")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--recipes", default="base")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--pairs", default="DNS:LDAP,SSDP:UDP")
    ap.add_argument("--new-class-block", type=int, default=1)
    ap.add_argument("--old-class-block", type=int, default=5)
    ap.add_argument("--decisions", default=None, help="decisions.json written by `decide`")
    ap.add_argument("--run", default=None)
    ap.add_argument("--decisions-commit", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"separability": separability, "probe": probe, "decide": decide, "policy": policy,
     "score-test": score_test}[a.cmd](a)


if __name__ == "__main__":
    main()
