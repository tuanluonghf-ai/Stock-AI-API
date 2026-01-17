from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "inception" / "ui"
UI_DATA_EXCLUDE = {"phrase_bank_vi.py"}  # static data; exclude from generator scan
UI_ORPHAN_EXCLUDE_DIRS = {
    UI_ROOT.resolve(),
    (REPO_ROOT / "ui").resolve(),
}
REMOVED_MODULES = (
    "narrative_output_pack",
    "narrative_governance",
    "phrase_bank_vi",
    "phrase_resolver",
)


def _iter_py_files(root: Path) -> Iterable[Path]:
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                yield Path(base) / name


def _scan_orphan_imports(root: Path) -> List[Tuple[Path, str]]:
    hits: List[Tuple[Path, str]] = []
    for path in _iter_py_files(root):
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(str(path.resolve()).startswith(str(ex)) for ex in UI_ORPHAN_EXCLUDE_DIRS):
            continue
        if "/_archive/" in path.as_posix() or path.name.endswith(".archive.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for mod in REMOVED_MODULES:
            if mod in text:
                hits.append((path, mod))
    return hits


def _scan_ui_patterns(regex: re.Pattern[str]) -> List[Tuple[Path, int, str]]:
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py_files(UI_ROOT):
        rel = str(path.as_posix())
        if "/_archive/" in rel or rel.endswith(".archive.py"):
            continue
        if path.name in UI_DATA_EXCLUDE:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                hits.append((path, idx, line.strip()))
    return hits


def _run(cmd: List[str], env: dict | None = None, timeout: int = 300) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return 1, str(exc)


def _print_gate(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"{name}: {status}"
    if detail:
        line += f" | {detail}"
    print(line)


def _print_hits(hits: List[Tuple[Path, int, str]], limit: int = 20) -> None:
    shown = hits[:limit]
    for path, line_no, line in shown:
        rel = path.relative_to(REPO_ROOT)
        print(f"- {rel}:{line_no} {line}")
    if len(hits) > limit:
        print(f"... ({len(hits) - limit} more)")


def _resolve_harness_cmd() -> List[str]:
    if (REPO_ROOT / "tools" / "regression_harness.py").exists():
        return [sys.executable, str(REPO_ROOT / "tools" / "regression_harness.py")]
    if (REPO_ROOT / "regression_harness.py").exists():
        return [sys.executable, str(REPO_ROOT / "regression_harness.py")]
    return [sys.executable, "-m", "tools.regression_harness"]


def main() -> None:
    overall_ok = True

    # 1) compileall
    code, out = _run([sys.executable, "-m", "compileall", "inception", "-q"])
    ok = code == 0
    _print_gate("compileall", ok)
    if not ok:
        overall_ok = False
        if out:
            print(out)

    # 2) import origin
    import_cmd = [
        sys.executable,
        "-c",
        (
            "import inception, os; "
            "p=inception.__file__; "
            "print(p); "
            "assert os.path.abspath(p).lower().find(os.path.abspath('.').lower())!=-1"
        ),
    ]
    code, out = _run(import_cmd)
    ok = code == 0
    _print_gate("import_origin", ok)
    if out:
        print(out)
    if not ok:
        overall_ok = False
        print("package shadowing detected; likely editable install from old path")

    # 3) orphan import scan
    orphan_hits = _scan_orphan_imports(REPO_ROOT)
    ok = len(orphan_hits) == 0
    _print_gate("orphan_imports", ok, f"hits={len(orphan_hits)}")
    if not ok:
        overall_ok = False
        for path, mod in orphan_hits[:20]:
            print(f"- {path}: {mod}")

    # 4) regression harness x2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    harness_cmd = _resolve_harness_cmd()
    for i in range(2):
        code, out = _run(harness_cmd, env=env, timeout=600)
        ok = code == 0 and "PASS overall" in out
        _print_gate(f"harness_run{i+1}", ok)
        if not ok:
            overall_ok = False
            if out:
                print(out)

    # 5) UI greps
    openai_re = re.compile(r"from\s+openai|OpenAI\(|chat\.completions|client\.responses|_call_openai", re.I)
    narrative_re = re.compile(r"build_comm_paragraph_v1\(|phrase bank|rotate/hash|premortem", re.I)
    dash_re = re.compile(r'return\s+"-"|\bor\s+"-"\b|\s"-"\s')

    hits = _scan_ui_patterns(openai_re)
    ok = len(hits) == 0
    _print_gate("ui_zero_llm", ok, f"hits={len(hits)}")
    if not ok:
        overall_ok = False
        _print_hits(hits)

    hits = _scan_ui_patterns(narrative_re)
    ok = len(hits) == 0
    _print_gate("ui_zero_generator", ok, f"hits={len(hits)}")
    if not ok:
        overall_ok = False
        _print_hits(hits)

    hits = _scan_ui_patterns(dash_re)
    ok = len(hits) == 0
    _print_gate("ui_no_ascii_dash", ok, f"hits={len(hits)}")
    if not ok:
        overall_ok = False
        _print_hits(hits)

    _print_gate("overall", overall_ok)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
