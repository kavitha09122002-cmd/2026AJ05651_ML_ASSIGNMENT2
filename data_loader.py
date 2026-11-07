"""Loading and light cleaning of the UCI Dry Bean dataset (UCI id 602).

The loader tries three sources in order, so it works on a laptop, on the BITS
Virtual Lab, or on a machine with a restricted network:

1. A file you downloaded yourself and put in ./data  (.xlsx / .csv / .txt)
2. The `ucimlrepo` package (fetches straight from the UCI repository)
3. The direct UCI zip download

Dataset page: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_ID = 602
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
TARGET_COL = "Class"


def _read_table(name: str, raw: bytes) -> pd.DataFrame | None:
    """Parse one file (by extension) into a DataFrame, or return None."""
    lower = name.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if lower.endswith((".csv", ".txt")):
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        return df
    return None


def _from_local(data_dir: Path) -> pd.DataFrame | None:
    if not data_dir.exists():
        return None
    for pattern in ("*.xlsx", "*.xls", "*.csv", "*.txt"):
        for path in sorted(data_dir.glob(pattern)):
            df = _read_table(path.name, path.read_bytes())
            if df is not None and TARGET_COL in df.columns:
                print(f"[data] loaded local file: {path}")
                return df
    return None


def _from_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=UCI_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        print("[data] loaded through ucimlrepo")
        return df
    except Exception as exc:  # network blocked, package missing, ...
        print(f"[data] ucimlrepo route failed: {exc}")
        return None


def _from_zip_url() -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for member in zf.namelist():
                df = _read_table(member, zf.read(member))
                if df is not None and TARGET_COL in df.columns:
                    print(f"[data] loaded {member} from the UCI zip")
                    return df
    except Exception as exc:
        print(f"[data] direct zip route failed: {exc}")
    return None


def load_raw(data_dir: str | Path = "data") -> pd.DataFrame:
    data_dir = Path(data_dir)
    for loader in (lambda: _from_local(data_dir), _from_ucimlrepo, _from_zip_url):
        df = loader()
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            return df
    raise RuntimeError(
        "Could not obtain the Dry Bean dataset automatically.\n"
        "Download 'dry+bean+dataset.zip' from "
        "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset in a browser, "
        f"extract Dry_Bean_Dataset.xlsx into the folder '{data_dir}/' and re-run."
    )


def load_clean(data_dir: str | Path = "data") -> tuple[pd.DataFrame, dict]:
    """Return the cleaned frame and a small dict describing what was done."""
    raw = load_raw(data_dir)
    info: dict = {"rows_raw": int(len(raw))}

    raw[TARGET_COL] = raw[TARGET_COL].astype(str).str.strip().str.title()
    feature_cols = [c for c in raw.columns if c != TARGET_COL]
    raw[feature_cols] = raw[feature_cols].apply(pd.to_numeric, errors="coerce")

    info["missing_cells"] = int(raw.isna().sum().sum())
    raw = raw.dropna().reset_index(drop=True)

    # Exact duplicate rows would leak between train and test, so remove them.
    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    info["duplicates_removed"] = int(before - len(raw))
    info["rows_clean"] = int(len(raw))
    info["n_features"] = len(feature_cols)
    return raw, info
