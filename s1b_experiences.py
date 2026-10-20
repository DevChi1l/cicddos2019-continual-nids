"""Stage 1b - Experience manifests (CI / CII) and the E0-frozen scaler.

Experience assignment decides WHICH ROWS arrive in each training round. It is a list of
row ids and nothing else; no embedding or scaling happens here except fitting the
frozen scaler. Embedding happens later, inside the model, on each batch.

Schedule file (JSON), classes chosen by YOU after reading the audit:
  {"experiences": [["DNS", "LDAP"], ["NTP", "SNMP"], ["SSDP", "UDP"]]}
BENIGN is implied; do not list it. Experience t's list = the NEW attack classes of E_t.

  ci : all train-benign rows go to E0 only (older benign reaches later rounds via replay).
  cii: train-benign is shuffled (seeded) and cut into k DISJOINT portions B_0..B_{k-1};
       E_t gets B_t. Rows are never copied between experiences, so "fresh benign" is real.

Attack rows: at most --attack-cap train rows per class, sampled with a seed.
Only the TRAIN split is used. calib / val / test are never put in an experience.

Scaler policy (ours, not the paper's per-experience scaler): signed-log1p, then
standardise using E0 TRAIN rows only and freeze. One scaler for every later round, so
replayed old rows and new rows share a scale. A feature that is constant in E0 gets
std 1.0 instead of ~0 so a later non-zero value is not blown straight to the clip;
the +-CLIP still bounds everything. Protocol vocabulary is also fit on E0 (0 = unseen).

Transformed duplicates: two distinct raw rows can become the SAME model input after the
transform (clip, float32, -0.0, unseen protocol). Protected rows whose model input equals a
training input are excluded from scoring for BOTH scenarios (union over the CI and CII
scalers), recorded per split and class with their cause in protected_exclusions.json, and
the remaining overlap must be exactly 0. Training rows and the scaler are not changed.

Usage:
  python s1b_experiences.py --schedule schedule.json --scenario cii --attack-cap 20000
  python s1b_experiences.py --suggest      # print per-split class counts, no output files
  python s1b_experiences.py --schedule all_labels.json --scenario cii --tag all12 ...
      --tag writes every Stage 1b output under <out>/<tag>/ (the parquet splits are only read),
      so a second schedule never overwrites the files an earlier run was built from.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import LABEL_COL, SEED, default_work_dir, load_json, save_json

CLIP = 5.0
SD_FLOOR = 1e-3
SCENARIOS = ("ci", "cii")
PROTECTED = ("calib", "val", "test")


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_manifest(train: pd.DataFrame, schedule: list[list[str]], scenario: str,
                   attack_cap: int, seed: int = SEED) -> dict[int, np.ndarray]:
    """Return {experience index: array of train _rid values}."""
    rng = np.random.default_rng(seed)
    k = len(schedule)
    seen = [c for exp in schedule for c in exp]
    assert len(seen) == len(set(seen)), "a class appears in two experiences"
    assert "BENIGN" not in seen, "BENIGN is implied; do not list it"
    missing = [c for c in seen if c not in set(train[LABEL_COL])]
    assert not missing, f"classes with no train rows: {missing}"

    benign = train.loc[train[LABEL_COL] == "BENIGN", "_rid"].to_numpy()
    benign = rng.permutation(benign)
    if scenario == "ci":
        benign_parts = [benign] + [np.array([], dtype="int64")] * (k - 1)
    else:
        benign_parts = np.array_split(benign, k)  # disjoint, near-equal

    out = {}
    for t, classes in enumerate(schedule):
        ids = [benign_parts[t]]
        for c in classes:
            cid = train.loc[train[LABEL_COL] == c, "_rid"].to_numpy()
            if len(cid) > attack_cap:
                cid = rng.choice(cid, size=attack_cap, replace=False)
            ids.append(cid)
        out[t] = np.sort(np.concatenate(ids)).astype("int64")
    # disjointness check: a row id may appear in at most one experience
    allids = np.concatenate(list(out.values()))
    assert len(allids) == len(np.unique(allids)), "row id assigned to two experiences"
    return out


def fit_scaler(e0: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> dict:
    X = slog(e0[num_cols].to_numpy("float64"))
    mu, sd = X.mean(axis=0), X.std(axis=0)
    const_in_e0 = [c for c, s in zip(num_cols, sd) if s < SD_FLOOR]
    sd = np.where(sd < SD_FLOOR, 1.0, sd)
    vocab = {c: {str(v): i + 1 for i, v in enumerate(sorted(e0[c].astype(int).unique().tolist()))}
             for c in cat_cols}
    return {"features_numeric": num_cols, "features_categorical": cat_cols,
            "mean": mu.tolist(), "std": sd.tolist(), "clip": CLIP,
            "const_in_e0": const_in_e0, "cat_vocab": vocab,
            "transform": "signed_log1p -> (x-mean)/std -> clip", "fit_on": "E0 train rows only"}


def cat_codes(df: pd.DataFrame, sc: dict) -> np.ndarray:
    cats = sc["features_categorical"]
    return (np.stack([df[c].astype(int).astype(str).map(sc["cat_vocab"][c]).fillna(0).astype(int).to_numpy()
                      for c in cats], axis=1).astype("int64") if cats else np.zeros((len(df), 0), "int64"))


def apply_scaler(df: pd.DataFrame, sc: dict, class_map: dict):
    """DataFrame -> (x_num float32, x_cat int64, y int64) using a frozen scaler."""
    X = slog(df[sc["features_numeric"]].to_numpy("float64"))
    X = np.clip((X - np.array(sc["mean"])) / np.array(sc["std"]), -sc["clip"], sc["clip"])
    y = df[LABEL_COL].map(class_map).to_numpy("int64")
    return X.astype("float32"), cat_codes(df, sc), y


# ---- transformed duplicates ------------------------------------------------
# Stage 1 guarantees distinct RAW feature vectors per split. The model, however, sees the
# transformed input, and each step below can map two distinct raw rows to one input:
#   signed_zero     -0.0 vs 0.0 (Stage 1's float64 hash tells them apart; the model cannot)
#   slog_f64        float64 rounding of log1p for huge values
#   standardize_f64 float64 rounding of (x - mean) / std
#   clip            values beyond +-CLIP on the same side become equal
#   float32         differences below float32 precision vanish (slog squeezes big values)
#   model_input     protocol values unseen in E0 all map to code 0
# A collision is attributed to the FIRST step where the two rows become equal.
STAGES = ["signed_zero", "slog_f64", "standardize_f64", "clip", "float32", "model_input"]


def _rowhash(*blocks) -> np.ndarray:
    A = np.hstack([np.asarray(b, "float64") for b in blocks]) + 0.0  # -0.0 -> +0.0
    return pd.util.hash_pandas_object(pd.DataFrame(A), index=False).to_numpy()


def stage_hashes(df: pd.DataFrame, sc: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """{step: row hash after that step}, plus raw X and standardized Z for attribution.
    'model_input' is exactly what apply_scaler feeds the model."""
    cats = sc["features_categorical"]
    X = df[sc["features_numeric"]].to_numpy("float64")
    C = df[cats].to_numpy("float64") if cats else np.zeros((len(df), 0))
    S = slog(X)
    Z = (S - np.array(sc["mean"])) / np.array(sc["std"])
    K = np.clip(Z, -sc["clip"], sc["clip"])
    F = K.astype("float32")
    h = {"signed_zero": _rowhash(X, C), "slog_f64": _rowhash(S, C), "standardize_f64": _rowhash(Z, C),
         "clip": _rowhash(K, C), "float32": _rowhash(F, C), "model_input": _rowhash(F, cat_codes(df, sc))}
    return h, X, Z


def scaled_hash(df, sc, class_map=None) -> np.ndarray:
    """Hash of the exact model input after scaling (+-0 treated as equal, like the model)."""
    return stage_hashes(df, sc)[0]["model_input"]


def transformed_collisions(tr: pd.DataFrame, prot: pd.DataFrame, sc: dict) -> tuple[np.ndarray, dict]:
    """Protected rows whose model input equals a training input; why, per step and feature."""
    th, Xt, Zt = stage_hashes(tr, sc)
    ph, Xp, Zp = stage_hashes(prot, sc)
    hit = np.isin(ph["model_input"], th["model_input"])
    first = np.full(len(prot), "", dtype=object)
    for s in STAGES:  # equality is kept by every later step, so the first hit is the cause
        first[(first == "") & np.isin(ph[s], th[s])] = s
    # one matching training row per collision, to name the features that differed
    look = pd.Series(np.arange(len(tr)), index=th["model_input"])
    look = look[~look.index.duplicated()]
    pi, ti = np.flatnonzero(hit), look.loc[ph["model_input"][hit]].to_numpy()
    differ = Xp[pi] != Xt[ti]  # -0.0 == 0.0 here, so signed-zero pairs differ nowhere
    clipped = differ & (np.abs(Zp[pi]) > sc["clip"]) & (np.abs(Zt[ti]) > sc["clip"]) & \
        (np.sign(Zp[pi]) == np.sign(Zt[ti]))
    num = sc["features_numeric"]
    by_feat = {num[j]: {"clip": int(clipped[:, j].sum()), "rounding": int((differ & ~clipped)[:, j].sum())}
               for j in np.flatnonzero(differ.any(axis=0))}
    lab_p, lab_t = prot[LABEL_COL].to_numpy()[pi], tr[LABEL_COL].to_numpy()[ti]
    diag = {"rows": int(len(prot)), "collisions": int(hit.sum()),
            "first_step": pd.Series(first[hit]).value_counts().to_dict(),
            "same_label_as_train": int((lab_p == lab_t).sum()),
            "different_label_from_train": int((lab_p != lab_t).sum()),
            "features_that_differed": dict(sorted(by_feat.items(), key=lambda kv: -sum(kv[1].values())))}
    return hit, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--scenario", choices=["ci", "cii"], default="cii")
    ap.add_argument("--attack-cap", type=int, default=20_000, help="max train rows per attack class")
    ap.add_argument("--suggest", action="store_true")
    ap.add_argument("--tag", default="", help="write outputs to <out>/<tag>/ (default: <out>/)")
    ap.add_argument("--max-excluded-share", type=float, default=0.05,
                    help="refuse if excluding transformed duplicates would remove more than this share "
                         "of any class in any protected split")
    ap.add_argument("--min-benign-protected", type=int, default=5000,
                    help="refuse if fewer benign val or test rows remain after exclusion")
    ap.add_argument("--min-rows", type=int, default=1000,
                    help="refuse if any scheduled class, or any experience's benign, has fewer train rows")
    a = ap.parse_args()
    out = default_work_dir() if a.out is None else Path(a.out)

    split_manifest = load_json(out / "split_manifest.json")
    if a.suggest or a.schedule is None:
        print(pd.DataFrame(split_manifest["counts_per_split"]).fillna(0).astype(int))
        print("\nPick classes per experience, write schedule.json, rerun with --schedule.")
        return

    feats = load_json(out / "features.json")
    schedule = load_json(Path(a.schedule))["experiences"]
    train = pd.read_parquet(out / "clean_train.parquet")

    # Both scenarios are built here (deterministic) so that the protected exclusions are the
    # same set of row ids whichever scenario is being written.
    plans = {}
    for sc_name in SCENARIOS:
        ids_s = build_manifest(train, schedule, sc_name, a.attack_cap)
        e0 = train[train["_rid"].isin(ids_s[0])]
        plans[sc_name] = (ids_s, fit_scaler(e0, feats["numeric"], feats["categorical"]))
    ids, scaler = plans[a.scenario]
    known = ["BENIGN"] + [c for e in schedule for c in e]
    dest = out / a.tag
    excl = protected_exclusions(out, train, plans, known, a, dest)

    sdir = dest / a.scenario
    sdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sdir / "experience_rids.npz", **{f"e{t}": v for t, v in ids.items()})
    save_json(scaler, sdir / "scaler.json")

    lab = train.set_index("_rid")[LABEL_COL]
    per_exp = {f"E{t}": lab.loc[v].value_counts().to_dict() for t, v in ids.items()}
    thin = {e: {k: n for k, n in cnt.items() if n < a.min_rows} for e, cnt in per_exp.items()}
    thin = {e: v for e, v in thin.items() if v}
    if a.scenario == "cii":
        thin.update({e: {"BENIGN": cnt.get("BENIGN", 0)} for e, cnt in per_exp.items()
                     if cnt.get("BENIGN", 0) < a.min_rows})
    if thin:
        raise SystemExit(f"too few train rows (< {a.min_rows}): {thin}")

    kept = {sp: {k: v["kept"] for k, v in c.items()} for sp, c in excl["by_split_class"].items()}
    save_json({"scenario": a.scenario, "schedule": schedule, "attack_cap": a.attack_cap,
               "seed": SEED, "counts_per_experience": per_exp,
               "scaler_const_in_e0": scaler["const_in_e0"],
               "protected_exclusions": {"file": "protected_exclusions.json",
                                        "rids_sha256": excl["rids_sha256"],
                                        "n_excluded": excl["n_excluded"],
                                        "kept_by_split_class": kept},
               "scaled_input_overlap_before_exclusion":
                   {sp: d["collisions"] for sp, d in excl["diagnosis"][a.scenario].items()},
               "scaled_input_overlap": excl["residual"][a.scenario],
               "protected": "calib/val/test are never inside any experience; excluded rows are "
                            "never scored"},
              sdir / "experience_manifest.json")

    print(f"Scenario {a.scenario}: {len(ids)} experiences")
    print(pd.DataFrame(per_exp).fillna(0).astype(int))
    print("Features constant in E0 (std floored to 1.0):", scaler["const_in_e0"])
    print("protected rows kept for scoring:\n", pd.DataFrame(kept).astype(int))
    print(f"Saved to {sdir}")


def protected_exclusions(out: Path, train: pd.DataFrame, plans: dict, known: list[str], a,
                         dest: Path | None = None) -> dict:
    """Protected rows whose model input equals a training input are excluded from scoring.

    A protected (calib/val/test) row is excluded if, under the frozen scaler of ANY scenario,
    its model input equals that of ANY row used for training in ANY scenario. The same row
    ids are then removed for CI and CII, so both are scored on identical rows. Only features
    are compared (no labels, no model, no metric), and training rows are left untouched.
    Writes protected_exclusions.json (counts per split and class, causes, residual check).
    Refuses if a split/class loses more than --max-excluded-share, if fewer than
    --min-benign-protected benign val/test rows remain, or if any residual overlap is left.
    """
    used = np.unique(np.concatenate([v for ids_s, _ in plans.values() for v in ids_s.values()]))
    tr = train[train["_rid"].isin(used)].reset_index(drop=True)
    prot = {}
    for sp in PROTECTED:
        d = pd.read_parquet(out / f"clean_{sp}.parquet")
        prot[sp] = d[d[LABEL_COL].isin(known)].reset_index(drop=True)

    excluded, diag = set(), {}
    for sc_name, (_, scaler) in plans.items():
        for sp, d in prot.items():
            hit, dg = transformed_collisions(tr, d, scaler)
            excluded |= set(d.loc[hit, "_rid"].tolist())
            diag.setdefault(sc_name, {})[sp] = dg
    ex = np.array(sorted(excluded), dtype="int64")

    by, residual = {}, {}
    for sp, d in prot.items():
        out_ = d["_rid"].isin(ex).to_numpy()
        by[sp] = {}
        for k in known:
            m = (d[LABEL_COL] == k).to_numpy()
            n, e = int(m.sum()), int((m & out_).sum())
            by[sp][k] = {"rows": n, "excluded": e, "kept": n - e, "share_excluded": e / max(n, 1)}
    for sc_name, (_, scaler) in plans.items():  # recheck: nothing left equals a training input
        th = scaled_hash(tr, scaler)
        residual[sc_name] = {}
        for sp, d in prot.items():
            left = d[~d["_rid"].isin(ex)]
            n = int(np.isin(scaled_hash(left, scaler), th).sum())
            residual[sc_name][sp] = {"rows": int(len(left)), "equal_to_a_train_input": n,
                                     "share": n / max(len(left), 1)}
    res = {"rule": " ".join(protected_exclusions.__doc__.split("\n\n")[1].split()),
           "schedule": load_json(Path(a.schedule))["experiences"], "attack_cap": a.attack_cap,
           "seed": SEED, "scenarios": list(plans), "n_training_rows_compared": int(len(tr)),
           "n_excluded": int(len(ex)),
           "rids_sha256": hashlib.sha256(ex.tobytes()).hexdigest(),
           "by_split_class": by, "residual": residual, "diagnosis": diag,
           "excluded_rids": ex.tolist()}
    save_json(res, (dest or out) / "protected_exclusions.json")  # saved before any refusal, for inspection

    print("transformed duplicates of training inputs (protected rows), per scenario scaler:")
    for sc_name, sp_d in diag.items():
        for sp, dg in sp_d.items():
            print(f"  {sc_name}/{sp}: {dg['collisions']}/{dg['rows']} | first step {dg['first_step']} | "
                  f"same label {dg['same_label_as_train']}, different {dg['different_label_from_train']}")
            top = list(dg["features_that_differed"].items())[:5]
            if top:
                print(f"      features that differed: {top}")
    print(f"excluded protected rows (union over {list(plans)}): {len(ex)}")
    print(pd.DataFrame({sp: {k: v["excluded"] for k, v in c.items()} for sp, c in by.items()}).astype(int))

    bad = [f"{sc}/{sp}: {r['equal_to_a_train_input']}" for sc, x in residual.items()
           for sp, r in x.items() if r["equal_to_a_train_input"]]
    if bad:
        raise SystemExit(f"BUG: protected rows still equal a training input after exclusion: {bad}")
    too_many = {f"{sp}/{k}": round(v["share_excluded"], 4) for sp, c in by.items()
                for k, v in c.items() if v["share_excluded"] > a.max_excluded_share}
    if too_many:
        raise SystemExit(f"refuse: exclusion would remove > {a.max_excluded_share:.0%} of {too_many}; "
                         "the transform is merging this class too much - inspect the diagnosis "
                         "in protected_exclusions.json instead of dropping rows.")
    ben = {sp: by[sp]["BENIGN"]["kept"] for sp in ("val", "test")}
    if min(ben.values()) < a.min_benign_protected:
        raise SystemExit(f"refuse: benign rows left for false-alarm measurement {ben} "
                         f"< {a.min_benign_protected}")
    return res


if __name__ == "__main__":
    main()
