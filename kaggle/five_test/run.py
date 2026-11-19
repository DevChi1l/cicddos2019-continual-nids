# Kaggle script kernel (CPU is fine, Internet ON). Scores FIVE's deployed models on held-out test ONCE.
import glob, shutil, subprocess, sys
from pathlib import Path

DECISIONS_COMMIT = "__FILL_ME__"
REPO = "https://github.com/DevChi1l/cicddos2019-continual-nids"
CODE = Path("/kaggle/working/code")
subprocess.run(["git", "clone", REPO, str(CODE)], check=True)
hits = sorted(glob.glob("/kaggle/input/**/nids_out/five/summary.json", recursive=True))
assert hits, "attach the five_val kernel output"
W = Path("/kaggle/working/nids_out")
if not W.exists():
    shutil.copytree(Path(hits[0]).parents[1], W, ignore=shutil.ignore_patterns("code"))
subprocess.run([sys.executable, "s16_five.py", "score-test", "--work", str(W), "--decisions-commit",
                DECISIONS_COMMIT, "--repo", str(CODE)], cwd=CODE, check=True)
shutil.copy2(W / "five" / "test.json", "/kaggle/working/five_test.json")
