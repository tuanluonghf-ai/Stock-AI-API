from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd

ZONE_TYPES = {
    "STRUCTURE_WALL": "STRUCTURE_WALL",
    "TECH_WALL": "TECH_WALL",
    "EXPECTATION_TARGET": "EXPECTATION_TARGET",
    "EXPECTATION_POTENTIAL": "EXPECTATION_POTENTIAL",
}


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _find_atr(ap: Dict[str, Any]) -> Optional[float]:
    keys = {"atr", "atr14", "atr_14", "atr14pct", "atr_pct", "atrp", "atrp14", "atrp_14"}
    direct = ap or {}
    for k, v in direct.items():
        if isinstance(k, str) and k.lower() in keys:
            val = _safe_float(v, default=math.nan)
            if math.isfinite(val):
                return float(val)

    for parent_key in ("Last", "Meta", "ProTech", "PriceAction", "CoreStats"):
        p = direct.get(parent_key)
        if isinstance(p, dict):
            for k, v in p.items():
                if isinstance(k, str) and k.lower() in keys:
                    val = _safe_float(v, default=math.nan)
                    if math.isfinite(val):
                        return float(val)
    return None


def _atr14_from_df(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return math.nan
    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            return math.nan
    dfx = df.copy()
    hi = pd.to_numeric(dfx["High"], errors="coerce")
    lo = pd.to_numeric(dfx["Low"], errors="coerce")
    close = pd.to_numeric(dfx["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    val = _safe_float(atr.iloc[-1], default=math.nan)
    return val


def _normalize_fib_key(k: Any) -> Optional[float]:
    if k is None:
        return None
    s = str(k).strip().lower().replace("%", "")
    s = s.replace("fib", "").replace("ratio", "").strip()
    s = s.replace(" ", "")
    try:
        v = float(s)
    except Exception:
        return None
    if v > 1.5:
        v = v / 100.0
    return float(round(v, 3))


def _fib_levels_from_pack(fib_pack: Dict[str, Any]) -> Tuple[Dict[float, float], Dict[float, float]]:
    short_lv: Dict[float, float] = {}
    long_lv: Dict[float, float] = {}
    if not isinstance(fib_pack, dict):
        return short_lv, long_lv

    def _extract(levels: Any) -> Dict[float, float]:
        out: Dict[float, float] = {}
        if not isinstance(levels, dict):
            return out
        for k, v in levels.items():
            ratio = _normalize_fib_key(k)
            if ratio is None:
                continue
            price = _safe_float(v, default=math.nan)
            if math.isfinite(price):
                out[ratio] = float(price)
        return out

    short_levels = _extract((fib_pack.get("Short") or {}).get("levels") if isinstance(fib_pack.get("Short"), dict) else {})
    long_levels = _extract((fib_pack.get("Long") or {}).get("levels") if isinstance(fib_pack.get("Long"), dict) else {})
    return short_levels, long_levels


def _recent_swing_levels(
    df_view: pd.DataFrame,
    lookback: int = 220,
    pivot: int = 10,
    skip_last: int = 12,
) -> Tuple[float, float]:
    if df_view is None or df_view.empty:
        return (math.nan, math.nan)
    if "High" not in df_view.columns or "Low" not in df_view.columns:
        return (math.nan, math.nan)

    if len(df_view) > skip_last + pivot * 2:
        dfz = df_view.iloc[:-skip_last]
    else:
        dfz = df_view

    dfx = dfz.tail(max(1, int(lookback))).copy()
    hi = pd.to_numeric(dfx["High"], errors="coerce")
    lo = pd.to_numeric(dfx["Low"], errors="coerce")

    swing_high = math.nan
    swing_low = math.nan
    start = pivot
    end = max(start, len(dfx) - pivot)
    for i in range(end - 1, start - 1, -1):
        win_hi = hi.iloc[i - pivot : i + pivot + 1]
        if not win_hi.dropna().empty and hi.iloc[i] == win_hi.max():
            swing_high = _safe_float(hi.iloc[i], default=math.nan)
            break

    for i in range(end - 1, start - 1, -1):
        win_lo = lo.iloc[i - pivot : i + pivot + 1]
        if not win_lo.dropna().empty and lo.iloc[i] == win_lo.min():
            swing_low = _safe_float(lo.iloc[i], default=math.nan)
            break

    if not math.isfinite(swing_high):
        swing_high = _safe_float(hi.max(), default=math.nan)
    if not math.isfinite(swing_low):
        swing_low = _safe_float(lo.min(), default=math.nan)
    return (float(swing_high), float(swing_low))


def compute_battle_map_pack_v1(df: Any, analysis_pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ap = analysis_pack or {}
    if not isinstance(ap, dict):
        ap = {}

    df_in = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df) if df is not None else pd.DataFrame()
    ref = _safe_float((df_in["Close"].iloc[-1] if not df_in.empty and "Close" in df_in.columns else ap.get("Last", {}).get("Close")), default=math.nan)
    if not math.isfinite(ref) or ref <= 0:
        ref = _safe_float(ap.get("Last", {}).get("Close"), default=math.nan)
    if not math.isfinite(ref) or ref <= 0:
        ref = _safe_float(pd.to_numeric(df_in.get("Close"), errors="coerce").dropna().iloc[-1] if "Close" in df_in.columns and not df_in.empty else math.nan, default=math.nan)

    atr = _find_atr(ap)
    atr_fallback = False
    if atr is None or not math.isfinite(float(atr)):
        atr = _atr14_from_df(df_in)
    if not math.isfinite(float(atr)) or float(atr) <= 0:
        atr = ref * 0.02 if math.isfinite(ref) else 0.0
        atr_fallback = True
    atr = float(atr)

    eps = max(atr * 0.6, ref * 0.008) if math.isfinite(ref) else atr * 0.6
    thickness = _clamp(atr * 0.8, ref * 0.006, ref * 0.02) if math.isfinite(ref) else atr * 0.8

    df_view = df_in.copy()
    if not df_view.empty and "Date" in df_view.columns:
        df_view["Date"] = pd.to_datetime(df_view["Date"], errors="coerce")
        df_view = df_view.dropna(subset=["Date"]).sort_values("Date")
    if not df_view.empty and "Close" in df_view.columns:
        close_full = pd.to_numeric(df_view["Close"], errors="coerce")
        last_n = 252
        close_1y = close_full.tail(last_n)
        if not close_1y.dropna().empty:
            last_close = close_1y.dropna().iloc[-1]
            rng = (close_1y.max() - close_1y.min()) / last_close if last_close else 0.0
            if rng < 0.18:
                last_n = min(360, 420)
        df_view = df_view.tail(min(last_n, 420)).copy()

    # MA candidates
    last = ap.get("Last") if isinstance(ap.get("Last"), dict) else {}
    ma_vals = {
        "MA20": _safe_float(last.get("MA20"), default=math.nan),
        "MA50": _safe_float(last.get("MA50"), default=math.nan),
        "MA200": _safe_float(last.get("MA200"), default=math.nan),
    }
    if (not math.isfinite(ma_vals["MA20"])) and "Close" in df_in.columns:
        ma_vals["MA20"] = _safe_float(pd.to_numeric(df_in["Close"], errors="coerce").rolling(20).mean().iloc[-1], default=math.nan)
    if (not math.isfinite(ma_vals["MA50"])) and "Close" in df_in.columns:
        ma_vals["MA50"] = _safe_float(pd.to_numeric(df_in["Close"], errors="coerce").rolling(50).mean().iloc[-1], default=math.nan)
    if (not math.isfinite(ma_vals["MA200"])) and "Close" in df_in.columns:
        ma_vals["MA200"] = _safe_float(pd.to_numeric(df_in["Close"], errors="coerce").rolling(200).mean().iloc[-1], default=math.nan)

    candidates: List[Dict[str, Any]] = []
    ma_weights = {"MA20": 1.0, "MA50": 2.0, "MA200": 3.0}
    for name, val in ma_vals.items():
        if math.isfinite(val):
            candidates.append(
                {
                    "price": float(val),
                    "weight": ma_weights[name],
                    "tags": [name],
                    "kind": "MA",
                    "zone_type": ZONE_TYPES["TECH_WALL"],
                }
            )

    fib_pack = ap.get("Fibonacci") if isinstance(ap.get("Fibonacci"), dict) else {}
    short_levels, long_levels = _fib_levels_from_pack(fib_pack)
    keep = {0.382, 0.5, 0.618, 0.786}
    if 0.236 in short_levels or 0.236 in long_levels:
        keep.add(0.236)

    for ratio, price in long_levels.items():
        if ratio in keep and math.isfinite(price):
            candidates.append(
                {
                    "price": float(price),
                    "weight": 2.2,
                    "tags": [f"FIB_LONG_{ratio}"],
                    "kind": "FIB_LONG",
                    "zone_type": ZONE_TYPES["TECH_WALL"],
                }
            )
    for ratio, price in short_levels.items():
        if ratio in keep and math.isfinite(price):
            candidates.append(
                {
                    "price": float(price),
                    "weight": 1.6,
                    "tags": [f"FIB_SHORT_{ratio}"],
                    "kind": "FIB_SHORT",
                    "zone_type": ZONE_TYPES["TECH_WALL"],
                }
            )

    ext_candidates: List[Dict[str, Any]] = []
    for ratio, price in long_levels.items():
        if ratio >= 1.0 and ratio not in keep and math.isfinite(price):
            zone_type = ZONE_TYPES["EXPECTATION_TARGET"] if math.isfinite(ref) and float(price) > float(ref) else ZONE_TYPES["EXPECTATION_POTENTIAL"]
            ext_candidates.append(
                {
                    "price": float(price),
                    "weight": 2.2,
                    "tags": [f"FIB_EXT_LONG_{ratio}"],
                    "kind": "FIB_EXT_LONG",
                    "zone_type": zone_type,
                }
            )
    for ratio, price in short_levels.items():
        if ratio >= 1.0 and ratio not in keep and math.isfinite(price):
            zone_type = ZONE_TYPES["EXPECTATION_TARGET"] if math.isfinite(ref) and float(price) > float(ref) else ZONE_TYPES["EXPECTATION_POTENTIAL"]
            ext_candidates.append(
                {
                    "price": float(price),
                    "weight": 1.6,
                    "tags": [f"FIB_EXT_SHORT_{ratio}"],
                    "kind": "FIB_EXT_SHORT",
                    "zone_type": zone_type,
                }
            )

    zone_type_priority = {
        ZONE_TYPES["STRUCTURE_WALL"]: 0,
        "STRUCTURE_EXTREME": 0,
        ZONE_TYPES["TECH_WALL"]: 1,
        ZONE_TYPES["EXPECTATION_TARGET"]: 2,
        ZONE_TYPES["EXPECTATION_POTENTIAL"]: 3,
    }

    def _pick_zone_type(types: List[str]) -> str:
        best = ZONE_TYPES["TECH_WALL"]
        best_pr = zone_type_priority.get(best, 9)
        for t in types:
            pr = zone_type_priority.get(t, 9)
            if pr < best_pr:
                best = t
                best_pr = pr
        return best

    def _apply_tier_caps(zone_type: str, tier: str) -> str:
        cap = None
        if zone_type == ZONE_TYPES["EXPECTATION_TARGET"]:
            cap = "C"
        elif zone_type == ZONE_TYPES["EXPECTATION_POTENTIAL"]:
            cap = "D"
        elif zone_type == ZONE_TYPES["TECH_WALL"]:
            cap = "B"
        if not cap:
            return tier
        order = {"A": 3, "B": 2, "C": 1, "D": 0}
        if order.get(tier, 0) > order.get(cap, 0):
            return cap
        return tier

    def _build_zones(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cands = [c for c in cands if math.isfinite(c["price"])]
        cands.sort(key=lambda x: x["price"])
        clusters: List[List[Dict[str, Any]]] = []
        for c in cands:
            placed = False
            for cl in clusters:
                if abs(c["price"] - cl[-1]["price"]) <= eps:
                    cl.append(c)
                    placed = True
                    break
            if not placed:
                clusters.append([c])

        out: List[Dict[str, Any]] = []
        for cl in clusters:
            wsum = sum(float(x["weight"]) for x in cl)
            if wsum <= 0:
                continue
            center = sum(float(x["price"]) * float(x["weight"]) for x in cl) / wsum
            low = float(center - thickness / 2.0)
            high = float(center + thickness / 2.0)
            tags = [t for x in cl for t in x.get("tags", [])]
            cluster_zone_type = _pick_zone_type([str(x.get("zone_type") or "") for x in cl if x.get("zone_type")])

            has_ma = any(x["kind"] == "MA" for x in cl)
            has_fib = any(x["kind"].startswith("FIB") for x in cl)
            has_long = any(x["kind"] == "FIB_LONG" for x in cl)
            has_short = any(x["kind"] == "FIB_SHORT" for x in cl)
            has_ext = any(x["kind"].startswith("FIB_EXT") for x in cl)

            score = float(wsum)
            if has_ma and has_fib:
                score += 1.2
                tags.append("+MA+FIB")
            if has_long and has_short:
                score += 0.8
                tags.append("+LONG+SHORT")

            if score >= 6.0:
                tier = "A"
            elif score >= 4.0:
                tier = "B"
            elif score >= 2.5:
                tier = "C"
            else:
                tier = "D"

            if math.isfinite(ref) and center >= ref:
                side = "RESISTANCE"
            else:
                side = "SUPPORT"

            zone_type = cluster_zone_type or ZONE_TYPES["TECH_WALL"]
            tier = _apply_tier_caps(zone_type, tier)

            dist = abs(center - ref) / ref if math.isfinite(ref) and ref != 0 else math.nan
            out.append(
                {
                    "side": side,
                    "zone_type": zone_type,
                    "low": float(low),
                    "high": float(high),
                    "center": float(center),
                    "score": float(score),
                    "tier": tier,
                    "reasons": tags,
                    "distance_pct": float(dist) if math.isfinite(dist) else math.nan,
                }
            )
        return out

    zones = _build_zones(candidates + ext_candidates)

    structure_zones: List[Dict[str, Any]] = []
    if not df_view.empty and "Close" in df_view.columns:
        close_series = pd.to_numeric(df_view["Close"], errors="coerce")
        if not close_series.dropna().empty:
            pos_hi = int(np.nanargmax(close_series.values))
            hi_px = _safe_float(close_series.iloc[pos_hi], default=math.nan)
            if math.isfinite(hi_px):
                center = float(hi_px)
                low = center - eps
                high = center + eps
                dist = abs(center - ref) / ref if math.isfinite(ref) and ref != 0 else math.nan
                zone = {
                    "side": "RESISTANCE",
                    "zone_type": "STRUCTURE_EXTREME",
                    "low": float(low),
                    "high": float(high),
                    "center": float(center),
                    "score": 4.0,
                    "tier": "B",
                    "reasons": ["STRUCTURE_1Y_HIGH_CLOSE"],
                    "distance_pct": float(dist) if math.isfinite(dist) else math.nan,
                    "pivot_ix": int(pos_hi),
                }
                if "Date" in df_view.columns:
                    try:
                        zone["pivot_date"] = str(pd.to_datetime(df_view["Date"].iloc[pos_hi]).date())
                    except Exception:
                        pass
                structure_zones.append(zone)
            pos_lo = int(np.nanargmin(close_series.values))
            lo_px = _safe_float(close_series.iloc[pos_lo], default=math.nan)
            if math.isfinite(lo_px):
                center = float(lo_px)
                low = center - eps
                high = center + eps
                dist = abs(center - ref) / ref if math.isfinite(ref) and ref != 0 else math.nan
                zone = {
                    "side": "SUPPORT",
                    "zone_type": "STRUCTURE_EXTREME",
                    "low": float(low),
                    "high": float(high),
                    "center": float(center),
                    "score": 4.0,
                    "tier": "B",
                    "reasons": ["STRUCTURE_1Y_LOW_CLOSE"],
                    "distance_pct": float(dist) if math.isfinite(dist) else math.nan,
                    "pivot_ix": int(pos_lo),
                }
                if "Date" in df_view.columns:
                    try:
                        zone["pivot_date"] = str(pd.to_datetime(df_view["Date"].iloc[pos_lo]).date())
                    except Exception:
                        pass
                structure_zones.append(zone)

    zones = zones + structure_zones

    tier_rank = {"A": 4, "B": 3, "C": 2, "D": 1}

    def _rank_key(z: Dict[str, Any]) -> Tuple[int, float, float]:
        tr = tier_rank.get(z.get("tier"), 0)
        dist = _safe_float(z.get("distance_pct"), default=math.nan)
        dist = dist if math.isfinite(dist) else 1e9
        return (-tr, -float(z.get("score", 0.0)), dist)

    resist = [z for z in zones if z.get("side") == "RESISTANCE"]
    supp = [z for z in zones if z.get("side") == "SUPPORT"]

    def _has_short_ext(z: Dict[str, Any]) -> bool:
        reasons = z.get("reasons") if isinstance(z.get("reasons"), list) else []
        return any(str(r).startswith("FIB_EXT_SHORT") for r in reasons)

    def _dedupe_near(z_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not math.isfinite(ref) or ref <= 0:
            return z_list
        z_list = sorted(z_list, key=lambda z: float(z.get("center", 0.0)))
        out: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(z_list):
            current = z_list[idx]
            group = [current]
            j = idx + 1
            while j < len(z_list):
                other = z_list[j]
                if abs(float(other.get("center", 0.0)) - float(current.get("center", 0.0))) / ref < 0.01:
                    group.append(other)
                    j += 1
                else:
                    break
            def _group_rank(z: Dict[str, Any]) -> Tuple[int, float, float]:
                pr = zone_type_priority.get(z.get("zone_type"), 9)
                score = float(z.get("score") or 0.0)
                dist = _safe_float(z.get("distance_pct"), default=math.nan)
                dist = dist if math.isfinite(dist) else 1e9
                return (pr, -score, dist)
            best = sorted(group, key=_group_rank)[0]
            out.append(best)
            idx = j
        return out

    def _order_within(z_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            z_list,
            key=lambda z: (
                _safe_float(z.get("distance_pct"), default=1e9),
                -float(z.get("score") or 0.0),
            ),
        )

    def _select_side(z_list: List[Dict[str, Any]], order_types: List[str], max_expectation: str) -> List[Dict[str, Any]]:
        picked: List[Dict[str, Any]] = []
        by_type = {t: _order_within([z for z in z_list if z.get("zone_type") == t]) for t in order_types}
        if by_type.get("STRUCTURE_EXTREME"):
            picked.append(by_type["STRUCTURE_EXTREME"].pop(0))
        elif by_type.get(ZONE_TYPES["STRUCTURE_WALL"]):
            picked.append(by_type[ZONE_TYPES["STRUCTURE_WALL"]].pop(0))
        exp_count = 0
        for t in order_types:
            for z in by_type.get(t, []):
                if len(picked) >= 3:
                    break
                if t == max_expectation:
                    if exp_count >= 1:
                        continue
                    exp_count += 1
                if z not in picked:
                    picked.append(z)
            if len(picked) >= 3:
                break
        return picked[:3]

    resist = _dedupe_near(resist)
    supp = _dedupe_near(supp)

    above_order = ["STRUCTURE_EXTREME", ZONE_TYPES["STRUCTURE_WALL"], ZONE_TYPES["TECH_WALL"], ZONE_TYPES["EXPECTATION_TARGET"]]
    below_order = ["STRUCTURE_EXTREME", ZONE_TYPES["STRUCTURE_WALL"], ZONE_TYPES["TECH_WALL"], ZONE_TYPES["EXPECTATION_POTENTIAL"]]
    resist_sel = _select_side(resist, above_order, ZONE_TYPES["EXPECTATION_TARGET"])
    supp_sel = _select_side(supp, below_order, ZONE_TYPES["EXPECTATION_POTENTIAL"])

    zones_selected = {"resistances": resist_sel, "supports": supp_sel}

    def _print_debug(selected: Dict[str, List[Dict[str, Any]]]) -> None:
        counts: Dict[str, int] = {}
        for side_key, z_list in selected.items():
            for z in z_list:
                zt = str(z.get("zone_type") or "")
                counts[zt] = counts.get(zt, 0) + 1
        print("battle_map zones_selected counts:", counts)
        for side_key, z_list in selected.items():
            for z in z_list:
                reasons = z.get("reasons") if isinstance(z.get("reasons"), list) else []
                print(
                    f"{side_key}: {z.get('zone_type')} | {float(z.get('center', 0.0)):.2f} | {z.get('tier')} | {reasons[:2]}"
                )

    symbol = str(ap.get("Symbol") or ap.get("symbol") or ap.get("ticker") or "").upper()
    if symbol == "HCM":
        _print_debug(zones_selected)

    pack = {
        "reference_price": float(ref) if math.isfinite(ref) else math.nan,
        "atr": float(atr),
        "future_space_ratio": 0.25,
        "eps": float(eps),
        "thickness": float(thickness),
        "zones": zones,
        "zones_selected": zones_selected,
    }
    if atr_fallback:
        pack["atr_fallback"] = True
    return pack
