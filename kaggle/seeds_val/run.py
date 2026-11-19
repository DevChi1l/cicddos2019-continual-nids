# Kaggle script kernel (GPU T4, Internet ON). Input: notebook 04 output. SEEDS validation (s17_seeds.py run).
import glob, subprocess, sys, time
from pathlib import Path

REPO = "https://github.com/DevChi1l/cicddos2019-continual-nids"
CODE = Path("/kaggle/working/code")
if not CODE.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(CODE)], check=True)
print("code commit:", subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout)
src = None
for p in sorted(glob.glob("/kaggle/input/**/clean_train.parquet", recursive=True)):
    if (Path(p).parent / "split_manifest.json").exists() and (Path(p).parent / "cii" / "scaler.json").exists():
        src = Path(p).parent
        break
assert src, "attach notebook 04's output (nids_out with clean_*.parquet and cii/)"
t0 = time.time()
subprocess.run([sys.executable, "s17_seeds.py", "run", "--src", str(src), "--work", "/kaggle/working/nids_out"],
               cwd=CODE, check=True)
print(f"done in {time.time() - t0:.0f}s")
