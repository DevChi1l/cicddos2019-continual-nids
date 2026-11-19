# Kaggle script kernel (GPU or CPU, Internet ON). Scores the CG deployed models on held-out test ONCE.
# Inputs: the cg_val kernel's output. DECISIONS_COMMIT must be the commit that holds
# experiments/results/11_confgate/summary.json (s15 refuses otherwise).
import glob, shutil, subprocess, sys
from pathlib import Path

DECISIONS_COMMIT = "__FILL_ME__"
REPO = "https://github.com/DevChi1l/cicddos2019-continual-nids"
CODE = Path("/kaggle/working/code")
subprocess.run(["git", "clone", REPO, str(CODE)], check=True)  # full clone: s15 reads the decisions commit
hits = sorted(glob.glob("/kaggle/input/**/nids_out/cg/summary.json", recursive=True))
assert hits, "attach the cg_val kernel output"
src = Path(hits[0]).parents[1]
W = Path("/kaggle/working/nids_out")
if not W.exists():
    shutil.copytree(src, W, ignore=shutil.ignore_patterns("code"))
subprocess.run([sys.executable, "s15_confgate.py", "score-test", "--work", str(W), "--decisions-commit",
                DECISIONS_COMMIT, "--repo", str(CODE)], cwd=CODE, check=True)
shutil.copy2(W / "cg" / "test.json", "/kaggle/working/cg_test.json")
