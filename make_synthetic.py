"""Tiny synthetic CICDDoS2019-STYLE files for smoke tests only. Not real data; any
number produced from these is a plumbing check, never a result.

Mimics the messy parts we expect to see (leading-space headers, 'Unnamed: 0', Flow ID,
IPs, ports, Timestamp, Inbound, 'DrDoS_*' labels, +-inf, exact duplicates, one
feature-identical/label-different pair, benign ~1%) so the pipeline is exercised on them.

  python make_synthetic.py --out ./synth_input
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["DrDoS_DNS", "DrDoS_LDAP", "DrDoS_NTP", "DrDoS_SNMP", "Syn", "UDP"]
FEATS = [f" Feat {i}" for i in range(12)]


def make_file(rng, day, n_per_class, n_benign, offset):
    rows = []
    for ci, lab in enumerate([*CLASSES, "BENIGN"]):
        n = n_benign if lab == "BENIGN" else n_per_class
        X = rng.lognormal(mean=ci * 0.3, sigma=1.0, size=(n, len(FEATS)))
        df = pd.DataFrame(X, columns=FEATS)
        df[" Protocol"] = rng.choice([6, 17], size=n)
        df[" Source Port"] = rng.integers(0, 65535, n)
        df[" Destination Port"] = rng.integers(0, 65535, n)
        df["Flow ID"] = [f"f{day}-{lab}-{i}" for i in range(n)]
        df[" Source IP"] = "10.0.0.1"
        df[" Destination IP"] = "10.0.0.2"
        t0 = pd.Timestamp(f"2019-{day}") + pd.Timedelta(hours=offset + ci)
        df[" Timestamp"] = (t0 + pd.to_timedelta(rng.integers(0, 3000, n), unit="s")).astype(str)
        df[" Inbound"] = 0 if lab == "BENIGN" else 1  # deliberate giveaway for the audit
        df[" Label"] = lab
        rows.append(df)
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
    df.insert(0, "Unnamed: 0", np.arange(len(df)))
    df.loc[3, FEATS[0]] = np.inf
    df.loc[7, FEATS[1]] = np.nan
    df = pd.concat([df, df.iloc[10:15]], ignore_index=True)  # exact duplicates
    heavy = df[(df[" Label"] == "Syn") & np.isfinite(df[FEATS]).all(axis=1)].iloc[:1]
    df = pd.concat([df, pd.concat([heavy] * 300)], ignore_index=True)  # one flow copied 300x
    srng = np.random.default_rng(99)  # same rows in every file -> cross-file duplicates
    shared = pd.DataFrame(srng.lognormal(size=(40, len(FEATS))), columns=FEATS)
    for col in df.columns:
        if col not in FEATS:
            shared[col] = df[col].iloc[:40].to_numpy()
    shared[" Label"], shared[" Inbound"], shared[" Protocol"] = "DrDoS_LDAP", 1, 17
    df = pd.concat([df, shared[df.columns]], ignore_index=True)
    c = df.iloc[20:21].copy()  # same features, different label -> conflict
    c[" Label"] = "UDP" if c[" Label"].iloc[0] != "UDP" else "Syn"
    return pd.concat([df, c, transformed_twins(df, day)], ignore_index=True)


def transformed_twins(df, day):
    """Pairs of Syn rows that are distinct raw rows but become the same model input after the
    transform: -0.0 vs 0.0, beyond the +-5 clip, below float32 precision, unseen protocols
    (LDAP rows for the last kind: a class outside E0, so its protocols stay unseen)."""
    fin = np.isfinite(df[FEATS]).all(axis=1)
    out = []
    for kind, lab, col, a, b in [("zero", "Syn", FEATS[4], 0.0, -0.0), ("clip", "Syn", FEATS[3], 1e30, 2e30),
                                 ("float32", "Syn", FEATS[2], None, None),
                                 ("vocab", "DrDoS_LDAP", " Protocol", 1, 2)]:
        base = df[(df[" Label"] == lab) & fin].iloc[1:9]
        x, y = base.copy(), base.copy()
        if kind == "float32":
            y[col] = x[col] * (1 + 1e-12)
        else:
            x[col], y[col] = a, b
        x["Flow ID"] = [f"tw{day}-{kind}-{i}a" for i in range(len(x))]
        y["Flow ID"] = [f"tw{day}-{kind}-{i}b" for i in range(len(y))]
        out += [x, y]
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synth_input")
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--n-benign", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for day, name, off in [("01-12", "day1_part1.csv", 0), ("01-12", "day1_part2.csv", 12),
                           ("03-11", "day2_part1.csv", 0)]:
        make_file(rng, day, a.n_per_class, a.n_benign, off).to_csv(out / name, index=False)
        print("wrote", out / name)


if __name__ == "__main__":
    main()
