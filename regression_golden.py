#!/usr/bin/env python3
"""
Golden Regression Baseline (Engine/Contracts-first)

Usage:
  # Create / refresh baselines
  python regression_golden.py --update --tickers VNM CII MSN

  # Compare current vs baselines
  python regression_golden.py --tickers VNM CII MSN

Notes:
- Always writes under <repo_root>/golden
- Never requires UI; relies on inception.core.pipeline.build_result
- Safe-by-default: if a ticker fails, it records an error snapshot and continues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------
# Utilities
# ----------------------------

def _repo_root() -> Path:
    # repo root = folder containing this script
    return Path(__file__).resolve().parent

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _safe_get(d: Any, path: List[str], default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN

def _round_num(x: Any, nd: int = 4):
    try:
        if _is_number(x):
            return round(float(x), nd)
    except Exception:
        pass
    return x

def _compact_text(s: Any, max_len: int = 280) -> str:
    if not isinstance(s, str):
        return ""
    t = " ".join(s.strip().split())
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t

def _dump_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

# ----------------------------
# Snapshot extraction
# ----------------------------

def _extract_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact, high-signal snapshot that is stable across UI refactors.

    Focus:
      - ContractPack
      - DashboardSummaryPack (headline signals)
      - TradePlanPack / DecisionPack (action + risk framing)
      - NarrativeDraftPack (DNA/Status/Plan + validators)
    """
    ap = result.get("AnalysisPack") if isinstance(result, dict) else {}
    dash = result.get("DashboardSummaryPack", {}) if isinstance(result, dict) else {}

    # Packs may exist under different locations depending on versions.
    # Prefer AnalysisPack attached packs, then Result-level packs.
    def pick_pack(name: str) -> Dict[str, Any]:
        p = {}
        if isinstance(ap, dict):
            p = ap.get(name) or {}
        if not isinstance(p, dict) or not p:
            p = result.get(name) or {}
        return p if isinstance(p, dict) else {}

    contract = pick_pack("ContractPack")
    ndp = pick_pack("NarrativeDraftPack")

    # Character module output sometimes holds TradePlan/Decision
    modules = result.get("Modules", {}) if isinstance(result, dict) else {}
    ch = modules.get("character", {}) if isinstance(modules, dict) else {}
    trade_plan = {}
    decision = {}

    # Common keys in your codebase: TradePlanPack / DecisionPack
    for k in ("TradePlanPack", "TradePlanPack.v1", "TradePlan"):
        trade_plan = ch.get(k) if isinstance(ch, dict) and isinstance(ch.get(k), dict) else trade_plan
        trade_plan = ap.get(k) if isinstance(ap, dict) and isinstance(ap.get(k), dict) else trade_plan
        trade_plan = result.get(k) if isinstance(result.get(k), dict) else trade_plan

    for k in ("DecisionPack", "DecisionPack.v1", "Decision"):
        decision = ch.get(k) if isinstance(ch, dict) and isinstance(ch.get(k), dict) else decision
        decision = ap.get(k) if isinstance(ap, dict) and isinstance(ap.get(k), dict) else decision
        decision = result.get(k) if isinstance(result.get(k), dict) else decision

    # Minimal normalization for high-signal fields
    snap = {
        "schema": "GoldenSnapshot.v1",
        "generated_at_utc": _utc_now(),
        "ticker": _safe_get(ap, ["Ticker"], _safe_get(ap, ["ticker"], "")) or result.get("Ticker", ""),
        "contract": {
            "ok": bool(contract.get("ok")) if isinstance(contract, dict) else False,
            "issues": (contract.get("issues") or [])[:12] if isinstance(contract, dict) else [],
        },
        "dashboard": {
            "scenario": _safe_get(dash, ["scenario_name"], _safe_get(dash, ["Scenario", "Name"], "")),
            "master_score_total": _round_num(_safe_get(dash, ["master_score_total"], _safe_get(dash, ["MasterScore", "Total"], None)), 3),
            "conviction_score": _round_num(_safe_get(dash, ["conviction_score"], _safe_get(dash, ["ConvictionScore"], None)), 3),
            "style_tilt": _safe_get(dash, ["style_tilt"], _safe_get(dash, ["StyleTilt"], "")),
            "class": _safe_get(dash, ["class"], _safe_get(dash, ["Class"], "")),
            "gate_status": _safe_get(dash, ["gate_status"], ""),
        },
        "trade_plan": {
            "name": _safe_get(trade_plan, ["PrimarySetup", "Name"], _safe_get(trade_plan, ["setup_name"], _safe_get(trade_plan, ["Name"], ""))),
            "rr": _round_num(_safe_get(trade_plan, ["PrimarySetup", "RR"], _safe_get(trade_plan, ["rr"], None)), 3),
            "risk_pct": _round_num(_safe_get(trade_plan, ["PrimarySetup", "RiskPct"], _safe_get(trade_plan, ["risk_pct"], None)), 3),
            "reward_pct": _round_num(_safe_get(trade_plan, ["PrimarySetup", "RewardPct"], _safe_get(trade_plan, ["reward_pct"], None)), 3),
            "plan_completeness": _safe_get(trade_plan, ["PlanCompleteness"], _safe_get(trade_plan, ["plan_completeness"], "")),
            "triggers": _safe_get(trade_plan, ["Triggers"], _safe_get(trade_plan, ["triggers"], {})) if isinstance(_safe_get(trade_plan, ["Triggers"], _safe_get(trade_plan, ["triggers"], {})), dict) else {},
            "portion_1": _round_num(_safe_get(trade_plan, ["Portion1"], _safe_get(trade_plan, ["portion_1"], None)), 3),
            "portion_2": _round_num(_safe_get(trade_plan, ["Portion2"], _safe_get(trade_plan, ["portion_2"], None)), 3),
        },
        "decision": {
            "action": _safe_get(decision, ["Action"], _safe_get(decision, ["action"], "")),
            "conviction": _round_num(_safe_get(decision, ["Conviction"], _safe_get(decision, ["conviction"], None)), 3),
            "size_hint": _safe_get(decision, ["SizeHint"], _safe_get(decision, ["size_hint"], "")),
            "guardrails": _safe_get(decision, ["Guardrails"], _safe_get(decision, ["guardrails"], "")),
        },
        "narrative": {
            "dna": {
                "line": _compact_text(_safe_get(ndp, ["dna", "line_draft"], "")),
                "ok": bool(_safe_get(ndp, ["dna", "validator", "ok"], True)),
                "reasons": (_safe_get(ndp, ["dna", "validator", "reasons"], []) or [])[:5],
            },
            "status": {
                "line": _compact_text(_safe_get(ndp, ["status", "line_draft"], "")),
                "ok": bool(_safe_get(ndp, ["status", "validator", "ok"], True)),
                "reasons": (_safe_get(ndp, ["status", "validator", "reasons"], []) or [])[:5],
            },
            "plan": {
                "line": _compact_text(_safe_get(ndp, ["plan", "line_draft"], "")),
                "ok": bool(_safe_get(ndp, ["plan", "validator", "ok"], True)),
                "reasons": (_safe_get(ndp, ["plan", "validator", "reasons"], []) or [])[:5],
            },
        },
        "meta": {
            "module_errors": (result.get("_ModuleErrors") or [])[:10],
            "has_error": bool(result.get("Error")),
            "error": _compact_text(result.get("Error", "")),
        },
    }
    return snap

# ----------------------------
# Diff
# ----------------------------

@dataclass
class DiffItem:
    path: str
    a: Any
    b: Any

def _diff(a: Any, b: Any, path: str = "") -> List[DiffItem]:
    out: List[DiffItem] = []
    if type(a) != type(b):
        out.append(DiffItem(path, a, b))
        return out
    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for k in sorted(a_keys | b_keys):
            p = f"{path}.{k}" if path else k
            if k not in a:
                out.append(DiffItem(p, "<MISSING>", b[k]))
            elif k not in b:
                out.append(DiffItem(p, a[k], "<MISSING>"))
            else:
                out.extend(_diff(a[k], b[k], p))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            out.append(DiffItem(path + ".[len]", len(a), len(b)))
        for i in range(min(len(a), len(b))):
            out.extend(_diff(a[i], b[i], f"{path}[{i}]"))
        return out
    if a != b:
        out.append(DiffItem(path, a, b))
    return out

# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--update", action="store_true", help="Write/refresh golden baselines")
    parser.add_argument("--mode", default="FLAT", choices=["FLAT", "HOLDING"])
    parser.add_argument("--avg-cost", type=float, default=None)
    parser.add_argument("--size", type=float, default=None, help="position_size_pct_nav")
    parser.add_argument("--data-dir", default=None, help="Optional data directory (overrides DataHub.from_env default)")
    parser.add_argument("--price-vol", default="Price_Vol.xlsx")
    parser.add_argument("--hsc-target", default="Tickers Target Price.xlsx")
    parser.add_argument("--schema-only", action="store_true", help="Only check schema/keys/types; ignore value drifts")
    parser.add_argument("--max-diff", type=int, default=50)
    args = parser.parse_args()

    repo = _repo_root()
    golden_dir = repo / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GOLDEN] repo_root = {repo}")
    print(f"[GOLDEN] out_dir   = {golden_dir}")
    print(f"[GOLDEN] mode      = {args.mode}")

    # Import pipeline lazily so script can still run in partial environments.
    try:
        from inception.core.pipeline import build_result  # type: ignore
    except Exception as e:
        print(f"[ERROR] Cannot import inception.core.pipeline.build_result: {e}")
        return 2

    enabled_modules = ["character"]  # baseline: high-signal, low drift
    failures = 0
    diffs_total = 0
    manifest = {
        "schema": "GoldenManifest.v1",
        "generated_at_utc": _utc_now(),
        "tickers": [t.strip().upper() for t in args.tickers],
        "mode": args.mode,
        "enabled_modules": enabled_modules,
        "price_vol": args.price_vol,
        "hsc_target": args.hsc_target,
    }

    for t in manifest["tickers"]:
        try:
            res = build_result(
                ticker=t,
                data_dir=args.data_dir,
                price_vol_path=args.price_vol,
                hsc_target_path=args.hsc_target,
                enabled_modules=enabled_modules,
                position_mode=args.mode,
                avg_cost=args.avg_cost,
                position_size_pct_nav=args.size,
            )
        except Exception as e:
            res = {"Error": f"build_result exception: {e}"}

        snap = _extract_snapshot(res if isinstance(res, dict) else {"Error": "Invalid result type"})

        out_path = golden_dir / f"{t}.flat.json"
        if args.update:
            _dump_json(out_path, snap)
            print(f"[WRITE] {out_path}")
            continue

        # Compare mode
        if not out_path.exists():
            print(f"[MISSING] {out_path} (run with --update first)")
            failures += 1
            continue

        base = _load_json(out_path)
        if args.schema_only:
            # remove value fields that drift often; keep only keys/types
            def strip_values(x: Any):
                if isinstance(x, dict):
                    return {k: strip_values(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [strip_values(v) for v in x]
                return type(x).__name__
            base_s = strip_values(base)
            snap_s = strip_values(snap)
            d = _diff(base_s, snap_s, "")
        else:
            d = _diff(base, snap, "")

        if d:
            failures += 1
            diffs_total += len(d)
            print(f"[DIFF] {t}: {len(d)} difference(s)")
            for item in d[: args.max_diff]:
                print(f"  - {item.path}: {item.a} -> {item.b}")
            if len(d) > args.max_diff:
                print(f"  ... ({len(d) - args.max_diff} more)")
        else:
            print(f"[OK] {t}: no differences")

    # Write manifest each time (helps tracking baseline context)
    man_path = golden_dir / "_manifest.json"
    if args.update:
        _dump_json(man_path, manifest)
        print(f"[WRITE] {man_path}")

    if args.update:
        print("[DONE] Baselines updated.")
        return 0

    if failures:
        print(f"[FAIL] tickers_failed={failures}, diffs_total={diffs_total}")
        return 1

    print("[PASS] All tickers matched baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
