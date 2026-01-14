from __future__ import annotations

import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASES_DIR = REPO_ROOT / "releases"


def _iter_files(root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", ".venv"}]
        for name in files:
            if name.endswith(".pyc"):
                continue
            yield Path(base) / name


def _add_glob(paths: Set[Path], pattern: str) -> None:
    for p in REPO_ROOT.glob(pattern):
        if p.is_file():
            paths.add(p)


def _collect_paths() -> List[Path]:
    paths: Set[Path] = set()

    for rel in ("inception/core", "inception/ui", "inception/infra"):
        root = REPO_ROOT / rel
        if root.exists():
            for p in _iter_files(root):
                paths.add(p)

    # Tools gates
    sg = REPO_ROOT / "tools" / "sanity_gate.py"
    if sg.exists():
        paths.add(sg)
    harness = REPO_ROOT / "tools" / "regression_harness.py"
    if harness.exists():
        paths.add(harness)
    else:
        root_harness = REPO_ROOT / "regression_harness.py"
        if root_harness.exists():
            paths.add(root_harness)

    # Golden artifacts
    _add_glob(paths, "golden/*.flat.json")
    _add_glob(paths, "golden/_manifest.json")

    # Wiring + readme
    app_py = REPO_ROOT / "app.py"
    if app_py.exists():
        paths.add(app_py)
    _add_glob(paths, "README_PHASE_*.txt")

    return sorted(paths)


def _zip_name() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"inception_phE_lock_{ts}.zip"


def main() -> None:
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RELEASES_DIR / _zip_name()

    paths = _collect_paths()
    if not paths:
        print("No files to include.")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            rel = p.relative_to(REPO_ROOT)
            zf.write(p, arcname=str(rel))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"zip: {zip_path}")
    print(f"files: {len(paths)}")
    print(f"size_mb: {size_mb:.2f}")


if __name__ == "__main__":
    main()
