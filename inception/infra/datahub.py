from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import functools
import pandas as pd

class DataError(Exception):
    """Raised when required local data files cannot be loaded."""
    pass

def resolve_data_path(path: str, base_dir: Path) -> str:
    """
    Resolve a relative data path under base_dir with light heuristics.
    Keeps behavior compatible with the legacy app.py resolver.
    """
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    # try under base_dir
    cand = (base_dir / p).resolve()
    if cand.exists():
        return str(cand)

    # case-insensitive match
    if cand.parent.exists():
        target = p.name.lower()
        for f in cand.parent.glob("*"):
            if f.name.lower() == target:
                return str(f.resolve())

    # fallback: search by filename anywhere under base_dir (bounded)
    target = p.name.lower()
    for f in base_dir.rglob(p.name):
        if f.name.lower() == target:
            return str(f.resolve())

    return str(cand)

def _file_mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0

@functools.lru_cache(maxsize=32)
def _read_excel_cached(resolved_path: str, mtime: float) -> pd.DataFrame:
    # mtime is part of cache key
    return pd.read_excel(resolved_path)

@dataclass(frozen=True)
class DataHub:
    data_dir: Path

    @staticmethod
    def from_env(default_dir: Optional[Path] = None) -> "DataHub":
        base = os.environ.get("INCEPTION_DATA_DIR")
        if base:
            return DataHub(Path(base).resolve())
        return DataHub((default_dir or Path.cwd()).resolve())

    def load_price_vol(self, path: str) -> pd.DataFrame:
        resolved = resolve_data_path(path, self.data_dir)
        mt = _file_mtime(resolved)
        try:
            df = _read_excel_cached(resolved, mt)
        except Exception as e:
            raise DataError(
                f"Không đọc được dữ liệu Price_Vol.xlsx. Path='{path}' | Resolved='{resolved}'. Lỗi: {e}. "
                f"Tip: đặt INCEPTION_DATA_DIR trỏ tới thư mục chứa data."
            ) from e
        df.columns = [str(c).strip().title() for c in df.columns]
        rename = {"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}
        df.rename(columns=rename, inplace=True)
        if "Date" not in df.columns or "Ticker" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
        return df

    def load_hsc_targets(self, path: str) -> pd.DataFrame:
        resolved = resolve_data_path(path, self.data_dir)
        mt = _file_mtime(resolved)
        try:
            df = _read_excel_cached(resolved, mt)
        except Exception as e:
            raise DataError(f"Không đọc được dữ liệu HSC targets. Path='{path}' | Resolved='{resolved}'. Lỗi: {e}") from e
        df.columns = [str(c).strip().title() for c in df.columns]
        if "Ticker" not in df.columns:
            return pd.DataFrame()
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        return df

    def load_ticker_names(self, path: str) -> pd.DataFrame:
        resolved = resolve_data_path(path, self.data_dir)
        mt = _file_mtime(resolved)
        try:
            df = _read_excel_cached(resolved, mt)
        except Exception as e:
            raise DataError(f"Không đọc được dữ liệu ticker names. Path='{path}' | Resolved='{resolved}'. Lỗi: {e}") from e
        df.columns = [str(c).strip().title() for c in df.columns]
        if "Ticker" not in df.columns:
            return pd.DataFrame()
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        return df
