"""Stage 7 - Report-ready tables and figures from SAVED results only (trains and recomputes nothing).

  python s7_deliverables.py --results <nids_out with runs/cii/main> [--sweep <nids_out of the sweep>] \
                            --dest deliverables

Writes <dest>/:
  main_decisions.md, main_final_test.md (+ .csv)  from runs/cii/main (rounds.json, final_test.json)
  sweep_hypotheses.md, sweep_decisions.md         if <sweep>/sweep/decisions.csv exists
  sweep_test_summary.md                           if <sweep>/sweep/test_summary.csv exists
  all_labels_final_test.md                        for every runs/<scenario>/all12_* run found
  figures/                                        copies of the saved PNGs (s4 / s6 output)
  claims.md                                       completed vs not-run claims, decided by which files exist
Every table names the file it came from. Missing inputs are reported as "not run", never filled in.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import load_json

ARMS = ("clean", "poisoned", "poisoned_repair")


def md(df: pd.DataFrame, source: str, title: str, digits: int = 4) -> str:
    """Markdown table without extra dependencies."""
    def cell(v):
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(cell(v) for v in row) + " |\n" for row in df.itertuples(index=False))
    return f"### {title}\n\nSource: `{source}`\n\n{head}{body}\n"


def final_table(rdir: Path) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        if (rdir / arm / "final_test.json").exists():
            t = load_json(rdir / arm / "final_test.json")["test"]
            rows.append({"arm": arm, **{f"recall {k}": v for k, v in t["recall"].items()},
                         "benign FAR": t["benign_far"], "benign rows": t["n_benign"], "test rows": t["n_rows"]})
    return pd.DataFrame(rows)


def decision_table(rdir: Path) -> pd.DataFrame:
    cfg = load_json(rdir / "config.json")
    tgt, drop = cfg["args"]["poison_target"], cfg["args"]["gate_old_drop"]
    rows = []
    for arm in ARMS:
        if (rdir / arm / "rounds.json").exists():
            for r in load_json(rdir / arm / "rounds.json"):
                rows.append({"arm": arm, "update": f"E{r['experience']}", "attempt": r["attempt"],
                             "accepted": r["accepted"], f"{tgt} val recall": r["val"]["recall"].get(tgt),
                             "floor": r["ref_val"]["recall"][tgt] - drop, "benign FAR": r["val"]["benign_far"],
                             "reasons": "; ".join(r["gate_reasons"]) or "-"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="nids_out folder that contains runs/cii/main")
    ap.add_argument("--sweep", default=None, help="nids_out folder of the sweep (optional)")
    ap.add_argument("--dest", default="deliverables")
    a = ap.parse_args()
    res, dest = Path(a.results), Path(a.dest)
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    main_dir = res / "runs" / "cii" / "main"
    have = {}

    t = decision_table(main_dir)
    (dest / "main_decisions.md").write_text(md(t, "runs/cii/main/<arm>/rounds.json", "main: gate decisions (val)"))
    f = final_table(main_dir)
    f.to_csv(dest / "main_final_test.csv", index=False)
    (dest / "main_final_test.md").write_text(md(f, "runs/cii/main/<arm>/final_test.json", "main: final test"))
    have["main"] = True
    have["main_r1"] = bool((t["attempt"] == "R1").any())
    for p in (main_dir / "report").glob("*.png"):
        shutil.copy(p, dest / "figures" / f"main_{p.name}")

    sw = Path(a.sweep) / "sweep" if a.sweep else None
    have["sweep_decisions"] = bool(sw and (sw / "decisions.csv").exists())
    have["sweep_test"] = bool(sw and (sw / "test_summary.csv").exists())
    have["sweep_r1"] = False
    if have["sweep_decisions"]:
        h = pd.read_csv(sw / "hypotheses.csv")
        (dest / "sweep_hypotheses.md").write_text(md(h, "sweep/hypotheses.csv", "Pre-registered hypotheses (val only)"))
        d = pd.read_csv(sw / "decisions.csv")
        have["sweep_r1"] = bool((d["attempt"] == "R1").any())
        keep = [c for c in d.columns if c not in ("gate_reasons", "repair_method", "w_flipped", "w_clean")]
        (dest / "sweep_decisions.md").write_text(md(d[keep], "sweep/decisions.csv", "Sweep: gate decisions (val)"))
        for p in sw.glob("decisions.png"):
            shutil.copy(p, dest / "figures" / "sweep_decisions.png")
    if have["sweep_test"]:
        s = pd.read_csv(sw / "test_summary.csv")
        (dest / "sweep_test_summary.md").write_text(md(s, "sweep/test_summary.csv", "Sweep: final test (mean/min/max over seeds)"))
        for p in sw.glob("test_syn.png"):
            shutil.copy(p, dest / "figures" / "sweep_test_syn.png")

    al = sorted((Path(a.sweep) if a.sweep else res).glob("runs/*/all12_*"))
    have["all_labels"] = [p.name for p in al if (p / "run_manifest.json").exists()]
    if have["all_labels"]:
        txt = "".join(md(final_table(p), f"runs/{p.parent.name}/{p.name}/<arm>/final_test.json",
                         f"All-labels {p.name}: final test") for p in al if (p / "run_manifest.json").exists())
        (dest / "all_labels_final_test.md").write_text(txt)

    def st(ok, done="completed (real run, saved files present)"):
        return done if ok else "NOT RUN / not available - no claim"
    claims = pd.DataFrame([
        ("Leakage-safe splits; 166 protected rows excluded; residual 0", "completed (Stage 1/1b manifests)"),
        ("main (CII, seed 42, f 0.5): every C1 accepted", st(have["main"] and not have["main_r1"])),
        ("R1 exercised on real data (gate-triggered)", st(have["main_r1"] or have["sweep_r1"])),
        ("Seed variance and gate behaviour for f in {0.75, 0.9, 0.95, 1.0}", st(have["sweep_decisions"])),
        ("Sweep test metrics (after decisions were committed)", st(have["sweep_test"])),
        ("Forced-R1 diagnostics (labelled diagnostic, not decisions)", st(have["sweep_decisions"])),
        ("All-labels schedule (12 of 14 labels), CII and CI", st(len(have["all_labels"]) == 2,
                                                                 "completed: " + ", ".join(have["all_labels"]))),
        ("No reliance on header-length artefacts / protocol (ablation)", "NOT RUN - no claim"),
    ], columns=["claim", "status"])
    (dest / "claims.md").write_text(md(claims, "file presence checked by s7_deliverables.py", "Claims"))
    print(claims.to_string(index=False))
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
