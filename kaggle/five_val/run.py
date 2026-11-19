# Kaggle script kernel (GPU T4, Internet ON). Input: notebook 04 output. Validation only (FIVE, s16_five.py).
import glob, subprocess, sys, time
from pathlib import Path

REPO = "https://github.com/DevChi1l/cicddos2019-continual-nids"
CODE = Path("/kaggle/working/code")
if not CODE.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(CODE)], check=True)
print("code commit:", subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout)
src = None
for p in sorted(glob.glob("/kaggle/input/**/clean_train.parquet", recursive=True)):
    if (Path(p).parent / "split_manifest.json").exists():
        src = Path(p).parent
        break
assert src, "attach notebook 04's output (nids_out with clean_*.parquet)"
t0 = time.time()
for cmd in ("prepare", "run"):
    r = subprocess.run([sys.executable, "s16_five.py", cmd, "--src", str(src), "--work", "/kaggle/working/nids_out"],
                       cwd=CODE)
    if r.returncode != 0:
        print(f"s16 {cmd} stopped (exit {r.returncode}); see nids_out/five/pilot.json", flush=True)
        break
print(f"done in {time.time() - t0:.0f}s")
