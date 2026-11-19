# Kaggle script kernel (GPU T4, Internet ON). Inputs: notebook 04 output (nids_out with Stage 1, cii/, runs/cii/...)
# and notebook 06 output (FU2 models or fu2_restore.zip). Writes to /kaggle/working:
#   dashboard_assets.json   (make_dashboard_assets.py: real deployed models + sampled test rows + PyTorch outputs)
#   nids_out/cg/summary.json and nids_out/runs/cii/cg_*   (s15_confgate.py run: validation only, no test)
import glob, os, subprocess, sys, time
from pathlib import Path

REPO = "https://github.com/DevChi1l/cicddos2019-continual-nids"
CODE = Path("/kaggle/working/code")
if not CODE.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(CODE)], check=True)
print("code commit:", subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout)

def find_src():
    for p in sorted(glob.glob("/kaggle/input/**/runs/cii/all12_cii_s42/config.json", recursive=True)):
        src = Path(p).parents[3]
        if (src / "clean_train.parquet").exists() and (src / "cii" / "scaler.json").exists():
            return src
    raise SystemExit("notebook 04 output (nids_out with runs/cii/all12_cii_s42 and clean_*.parquet) not attached")

SRC = find_src()
print("src:", SRC)
os.environ["NIDS_FU2_SRC"] = str(SRC)
os.environ["NIDS_PREPARED"] = str(SRC)
t0 = time.time()
r = subprocess.run([sys.executable, "make_dashboard_assets.py", "--out", "/kaggle/working/dashboard_assets.json"],
                   cwd=CODE)
if r.returncode != 0:
    print("assets: a model was missing; rebuilding it with the committed code", flush=True)
    subprocess.run([sys.executable, "make_dashboard_assets.py", "--out", "/kaggle/working/dashboard_assets.json",
                    "--rebuild"], cwd=CODE, check=True)
print(f"assets done in {time.time() - t0:.0f}s", flush=True)
subprocess.run([sys.executable, "s15_confgate.py", "run", "--src", str(SRC), "--work", "/kaggle/working/nids_out"],
               cwd=CODE, check=True)
print(f"all done in {time.time() - t0:.0f}s")
