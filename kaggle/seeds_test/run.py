# Kaggle script kernel (CPU is fine, Internet ON). Scores SEEDS deployed models on held-out test ONCE.
import glob, shutil, subprocess, sys
from pathlib import Path

DECISIONS_COMMIT = "309ac649d356d22aabd1034ac0c84f9ecea9a5c9"
REPO = "https://github.com/DevChi1l/cicddos2019-continual-nids"
CODE = Path("/kaggle/working/code")
subprocess.run(["git", "clone", REPO, str(CODE)], check=True)
hits = sorted(glob.glob("/kaggle/input/**/nids_out/seeds/summary.json", recursive=True))
assert hits, "attach the seeds_val kernel output"
W = Path("/kaggle/working/nids_out")
if not W.exists():
    shutil.copytree(Path(hits[0]).parents[1], W, ignore=shutil.ignore_patterns("code"))
subprocess.run([sys.executable, "s17_seeds.py", "score-test", "--work", str(W), "--decisions-commit",
                DECISIONS_COMMIT, "--repo", str(CODE)], cwd=CODE, check=True)
shutil.copy2(W / "seeds" / "test.json", "/kaggle/working/seeds_test.json")
