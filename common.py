"""Shared helpers for the NIDS pipeline (Stages 0-2).

Nothing here assumes a specific Kaggle dataset path: files are discovered by
extension, and every assumption about CICDDoS2019 (column names, label names)
is checked by s0_audit.py and written to a manifest before anything is trained.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

SEED = 42

# ---- Column handling -------------------------------------------------------
# Identifier / metadata columns that must never be model inputs.
# Names are compared after normalisation (strip + lowercase).
ID_COLS = {
    "flow id", "source ip", "destination ip", "unnamed: 0", "unnamed: 0.1",
    "simillarhttp", "similarhttp", "timestamp",
}
# Ports are dropped by default: in reflection attacks the source port is a
# near-perfect label giveaway (e.g. DNS reflection comes from port 53), which
# would make the benchmark trivially easy. Set to False only as a deliberate,
# documented ablation.
DROP_PORTS = True
PORT_COLS = {"source port", "destination port"}
# Low-cardinality integer columns embedded with nn.Embedding (TabTransformer
# style) instead of being treated as continuous numbers.
CATEGORICAL_COLS = ["protocol"]
# "Inbound" (direction flag) is reported by some CICDDoS2019 copies and may be a
# near-perfect label giveaway. Dropped by default until s0_audit's
# inbound-vs-label table shows otherwise; flip only as a labelled ablation.
DROP_INBOUND = True
LEAK_COLS = {"inbound"}
LABEL_COL = "label"
TIME_COL = "timestamp"

# ---- Label handling --------------------------------------------------------
# Raw label spellings vary between copies of the dataset ("DrDoS_DNS", "Syn",
# "UDP-lag", "UDPLag", ...). Everything is mapped to one canonical name; labels
# not in this table are kept verbatim and flagged by the audit.
_CANON = {
    "benign": "BENIGN", "dns": "DNS", "ldap": "LDAP", "mssql": "MSSQL",
    "netbios": "NetBIOS", "ntp": "NTP", "snmp": "SNMP", "ssdp": "SSDP",
    "udp": "UDP", "udp-lag": "UDPLag", "udplag": "UDPLag", "syn": "Syn",
    "tftp": "TFTP", "portmap": "Portmap", "webddos": "WebDDoS",
    "portscan": "PortScan",
}


def norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def canon_label(x: str) -> str:
    s = str(x).strip()
    key = s.lower()
    if key.startswith("drdos_"):
        key = key[len("drdos_"):]
    return _CANON.get(key, s)  # unknown labels survive verbatim -> audit flags


def is_known_label(canon: str) -> bool:
    return canon in set(_CANON.values())


# ---- File discovery / reading ---------------------------------------------
def find_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = [p for ext in ("*.csv", "*.parquet") for p in root.rglob(ext)]
    return sorted(files)


def iter_chunks(path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Yield DataFrames with normalised column names, for CSV or parquet."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize):
            df = batch.to_pandas()
            df.columns = [norm_col(c) for c in df.columns]
            yield df
    else:
        for df in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                              skipinitialspace=True):
            df.columns = [norm_col(c) for c in df.columns]
            yield df


def sha256_file(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def default_work_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/nids_out")
    return Path(os.environ.get("NIDS_WORK", "./nids_out"))
