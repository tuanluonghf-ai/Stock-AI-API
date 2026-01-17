"""Validation helpers (non-raising by default).

These checks are intended for diagnostics and early detection of pack drift.
They should not crash the app in production.

Step 9 adds a small schema/required-path registry and an issue collector so
Executive Snapshot can surface warnings/errors without breaking rendering.

Design rules:
- Streamlit-free.
- Non-raising by default.
- Defensive about non-dict inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

from .contracts import as_dict


def _get_path(root: Any, path: Sequence[str]) -> Tuple[bool, Any]:
    cur = root
    for p in path:
        if not isinstance(cur, dict):
            return False, None
        if p not in cur:
            return False, None
        cur = cur.get(p)
    return True, cur


def validate_required_paths(pack: Any, paths: List[Sequence[str]]) -> List[str]:
    """Return list of missing path strings. Non-raising."""
    p = as_dict(pack)
    missing: List[str] = []
    for path in paths:
        ok, _ = _get_path(p, path)
        if not ok:
            missing.append(".".join(path))
    return missing


def validate_analysis_pack(ap: Any) -> Dict[str, Any]:
    """Validate a normalized analysis pack. Returns diagnostics."""
    apd = as_dict(ap)
    required = [
        ["Ticker"],
        ["Last", "Close"],
        ["PrimarySetup"],
        ["MasterScore", "Total"],
    ]
    missing = validate_required_paths(apd, required)
    return {"ok": len(missing) == 0, "missing": missing}


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, tuple, dict, set)):
        return len(v) == 0
    return False


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return v == v


def _first_key(d: Dict[str, Any], keys: Sequence[str]) -> Tuple[bool, Any, str]:
    for k in keys:
        if k in d:
            return True, d.get(k), k
    return False, None, ""


def _add_issue(
    issues: List[Dict[str, Any]],
    severity: str,
    path: str,
    message: str,
    issue_type: str,
    value: Any = None,
) -> None:
    it = {
        "severity": severity,
        "path": path,
        "message": message,
        "type": issue_type,
    }
    if value is not None:
        it["value_type"] = type(value).__name__
    issues.append(it)


def collect_data_quality_pack(
    analysis_pack: Any,
    character_pack: Any,
    extra: Optional[Dict[str, Any]] = None,
    require_investor_mapping: bool = False,
    max_issues: int = 10,
) -> Dict[str, Any]:
    """Return a JSON-safe DataQualityPack.v1.

    The goal is high-signal early alerts, not exhaustive validation.
    """
    ap = as_dict(analysis_pack)
    cp = as_dict(character_pack)

    issues: List[Dict[str, Any]] = []

    # --- ERROR: core runtime / dashboard must-have ---
    core_checks = [
        ("AnalysisPack.Ticker", ["Ticker"], str, "Ticker missing/empty"),
        ("AnalysisPack.Last.Close", ["Last", "Close"], (int, float), "Last.Close missing"),
        ("AnalysisPack.MasterScore.Total", ["MasterScore", "Total"], (int, float), "MasterScore.Total missing"),
        ("AnalysisPack.Conviction", ["Conviction"], (int, float), "Conviction missing"),
        ("AnalysisPack.PrimarySetup.Name", ["PrimarySetup", "Name"], str, "PrimarySetup.Name missing"),
    ]

    for full_path, path, exp_t, msg in core_checks:
        ok, v = _get_path(ap, path)
        if not ok:
            _add_issue(issues, "ERROR", full_path, msg, "MISSING")
            continue
        if exp_t is not None and v is not None and not isinstance(v, exp_t):
            _add_issue(issues, "ERROR", full_path, msg, "TYPE", value=v)
            continue
        if isinstance(v, str) and _is_empty(v):
            _add_issue(issues, "ERROR", full_path, msg, "EMPTY")

    # --- WARN: completeness (safe to run but should be visible) ---
    warn_checks = [
        ("AnalysisPack.Scenario12.Name", ["Scenario12", "Name"], str, "Scenario12.Name missing"),
        ("AnalysisPack.ProTech.MA.Regime", ["ProTech", "MA", "Regime"], str, "ProTech.MA.Regime missing"),
        ("AnalysisPack.ProTech.Volume.Regime", ["ProTech", "Volume", "Regime"], str, "ProTech.Volume.Regime missing"),
        ("AnalysisPack.Fibonacci", ["Fibonacci"], dict, "Fibonacci pack missing"),
        ("AnalysisPack.Fibonacci.ShortWindow.Band", ["Fibonacci", "ShortWindow", "Band"], str, "Fib short band missing"),
        ("AnalysisPack.Fibonacci.LongWindow.Band", ["Fibonacci", "LongWindow", "Band"], str, "Fib long band missing"),
        ("CharacterPack.ClassName", ["ClassName"], str, "ClassName missing"),
        ("CharacterPack.Conviction", ["Conviction"], dict, "Conviction pack missing"),
    ]

    for full_path, path, exp_t, msg in warn_checks:
        root = ap if full_path.startswith("AnalysisPack") else cp
        ok, v = _get_path(root, path)
        if not ok:
            _add_issue(issues, "WARN", full_path, msg, "MISSING")
            continue
        if exp_t is not None and v is not None and not isinstance(v, exp_t):
            _add_issue(issues, "WARN", full_path, msg, "TYPE", value=v)
            continue
        if (isinstance(v, (dict, list)) and _is_empty(v)) or (isinstance(v, str) and _is_empty(v)):
            _add_issue(issues, "WARN", full_path, msg, "EMPTY")

    # Extra packs: only check schema key existence (keep minimal)
    if isinstance(extra, dict):
        for name, pack in extra.items():
            pdict = as_dict(pack)
            if not isinstance(pdict.get("schema"), str) or _is_empty(pdict.get("schema")):
                _add_issue(issues, "WARN", f"{name}.schema", "schema missing", "MISSING")

    # InvestorMappingPack contract validation (facts-first)
    inv_path = "AnalysisPack.InvestorMappingPack"
    inv_pack = ap.get("InvestorMappingPack")
    if isinstance(inv_pack, dict):
        _validate_investor_mapping_pack(inv_pack, issues, inv_path)
    elif require_investor_mapping:
        _add_issue(issues, "ERROR", inv_path, "InvestorMappingPack missing", "MISSING")

    # deterministic ordering: ERROR first then WARN, then by path
    def _k(i: Dict[str, Any]) -> Tuple[int, str]:
        sev = i.get("severity") or "WARN"
        rank = 0 if sev == "ERROR" else 1
        return rank, str(i.get("path") or "")

    issues_sorted = sorted(issues, key=_k)

    err = sum(1 for x in issues_sorted if x.get("severity") == "ERROR")
    warn = sum(1 for x in issues_sorted if x.get("severity") == "WARN")

    return {
        "schema": "DataQualityPack.v1",
        "error_count": int(err),
        "warn_count": int(warn),
        "issues": issues_sorted[: max(0, int(max_issues))],
    }


def _validate_investor_mapping_pack(pack: Dict[str, Any], issues: List[Dict[str, Any]], base_path: str) -> None:
    meta_ok, meta, meta_key = _first_key(pack, ["Meta", "meta"])
    if not meta_ok or not isinstance(meta, dict):
        _add_issue(issues, "ERROR", f"{base_path}.Meta", "Meta missing", "MISSING")
        return

    labels_ok, labels, _ = _first_key(meta, ["labels", "Labels"])
    if not labels_ok or not isinstance(labels, dict):
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels", "labels missing", "MISSING")
        return

    match_ok, match_min, _ = _first_key(labels, ["match_min", "matchMin", "MatchMin"])
    if not match_ok:
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels.match_min", "match_min missing", "MISSING")
    elif not _is_number(match_min):
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels.match_min", "match_min type", "TYPE", value=match_min)
    elif abs(float(match_min) - 7.5) > 1e-9:
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels.match_min", "match_min value", "VALUE", value=match_min)

    partial_ok, partial_min, _ = _first_key(labels, ["partial_min", "partialMin", "PartialMin"])
    if not partial_ok:
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels.partial_min", "partial_min missing", "MISSING")
    elif not _is_number(partial_min):
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels.partial_min", "partial_min type", "TYPE", value=partial_min)
    elif abs(float(partial_min) - 4.5) > 1e-9:
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels.partial_min", "partial_min value", "VALUE", value=partial_min)

    if _is_number(match_min) and _is_number(partial_min):
        if float(match_min) <= float(partial_min):
            _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.labels", "match_min <= partial_min", "VALUE")

    scale_ok, scale, _ = _first_key(meta, ["scale", "Scale"])
    if not scale_ok:
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.scale", "scale missing", "MISSING")
    elif not isinstance(scale, str):
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.scale", "scale type", "TYPE", value=scale)
    elif scale != "0-10":
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.scale", "scale value", "VALUE", value=scale)

    version_ok, version, _ = _first_key(meta, ["version", "Version"])
    if not version_ok:
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.version", "version missing", "MISSING")
    elif not isinstance(version, str):
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.version", "version type", "TYPE", value=version)
    elif version != "v1":
        _add_issue(issues, "ERROR", f"{base_path}.{meta_key}.version", "version value", "VALUE", value=version)

    pent_ok, pent, _ = _first_key(pack, ["Pentagon", "pentagon"])
    if not pent_ok:
        _add_issue(issues, "ERROR", f"{base_path}.Pentagon", "Pentagon missing", "MISSING")
    elif not isinstance(pent, dict):
        _add_issue(issues, "ERROR", f"{base_path}.Pentagon", "Pentagon type", "TYPE", value=pent)
    else:
        axes = {
            "TrendPower": ["TrendPower", "Trend Power"],
            "Explosive": ["Explosive"],
            "SafetyShield": ["SafetyShield", "Safety Shield"],
            "TradingFlow": ["TradingFlow", "Trading Flow"],
            "Adrenaline": ["Adrenaline"],
        }
        for axis, keys in axes.items():
            ok, val, key = _first_key(pent, keys)
            if not ok:
                _add_issue(issues, "ERROR", f"{base_path}.Pentagon.{axis}", "axis missing", "MISSING")
                continue
            if not _is_number(val):
                _add_issue(issues, "ERROR", f"{base_path}.Pentagon.{key}", "axis type", "TYPE", value=val)
                continue
            if float(val) < 0.0 or float(val) > 10.0:
                _add_issue(issues, "ERROR", f"{base_path}.Pentagon.{key}", "axis range", "RANGE", value=val)

    def _check_persona_item(item: Dict[str, Any], path: str, persona_name: str | None = None) -> None:
        name = None
        if isinstance(item, dict):
            ok, name_val, _ = _first_key(item, ["name", "Name", "Persona"])
            if ok and isinstance(name_val, str):
                name = name_val
        if not name and persona_name:
            name = persona_name
        if not name:
            _add_issue(issues, "ERROR", f"{path}.name", "persona name missing", "MISSING")

        score_ok, score, _ = _first_key(item, ["score_10", "Score10", "score"])
        if score_ok:
            if not _is_number(score):
                _add_issue(issues, "ERROR", f"{path}.score", "score type", "TYPE", value=score)
            elif float(score) < 0.0 or float(score) > 10.0:
                _add_issue(issues, "ERROR", f"{path}.score", "score range", "RANGE", value=score)

        label_ok, label, _ = _first_key(item, ["label", "Label"])
        if label_ok:
            if not isinstance(label, str):
                _add_issue(issues, "ERROR", f"{path}.label", "label type", "TYPE", value=label)
            elif label not in {"Match", "Partial", "Partial Match", "Mismatch"}:
                _add_issue(issues, "ERROR", f"{path}.label", "label value", "VALUE", value=label)

        if label_ok and score_ok and _is_number(score) and isinstance(label, str):
            hi = 7.5
            lo = 4.5
            if _is_number(match_min) and _is_number(partial_min):
                hi = float(match_min)
                lo = float(partial_min)
            score_val = float(score)
            label_norm = "Partial" if label == "Partial Match" else label
            expected = "Match" if score_val >= hi else "Partial" if score_val >= lo else "Mismatch"
            if label_norm != expected:
                _add_issue(issues, "WARN", f"{path}.label", "label mismatch vs score", "MISMATCH", value=label)

        veto_ok, vetoed, _ = _first_key(item, ["vetoed", "Vetoed"])
        if veto_ok and not isinstance(vetoed, bool):
            _add_issue(issues, "ERROR", f"{path}.vetoed", "vetoed type", "TYPE", value=vetoed)

        for key in ("reasons", "tags", "Reasons", "Tags"):
            if key in item:
                val = item.get(key)
                if not isinstance(val, list) or any(not isinstance(x, str) for x in val):
                    _add_issue(issues, "ERROR", f"{path}.{key}", "list[str] required", "TYPE", value=val)

    personas_key = None
    personas_val = None
    for k in ("Personas", "personas", "Compatibility", "compatibility", "PersonaMatch", "persona_match"):
        if k in pack:
            personas_key = k
            personas_val = pack.get(k)
            break

    if personas_key is not None:
        ppath = f"{base_path}.{personas_key}"
        if isinstance(personas_val, list):
            for i, item in enumerate(personas_val):
                if not isinstance(item, dict):
                    _add_issue(issues, "ERROR", f"{ppath}[{i}]", "persona item type", "TYPE", value=item)
                    continue
                _check_persona_item(item, f"{ppath}[{i}]")
        elif isinstance(personas_val, dict):
            for key, item in personas_val.items():
                if not isinstance(item, dict):
                    _add_issue(issues, "ERROR", f"{ppath}.{key}", "persona item type", "TYPE", value=item)
                    continue
                _check_persona_item(item, f"{ppath}.{key}", persona_name=str(key))
        else:
            _add_issue(issues, "ERROR", ppath, "personas type", "TYPE", value=personas_val)

    if "vetoed" in pack and not isinstance(pack.get("vetoed"), bool):
        _add_issue(issues, "ERROR", f"{base_path}.vetoed", "vetoed type", "TYPE", value=pack.get("vetoed"))
    for key in ("reasons", "tags"):
        if key in pack:
            val = pack.get(key)
            if not isinstance(val, list) or any(not isinstance(x, str) for x in val):
                _add_issue(issues, "ERROR", f"{base_path}.{key}", "list[str] required", "TYPE", value=val)

# ---------------------------------------------------------------------
# Narrative validation (DNA tone guardrails)
# ---------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:[\.,]\d+)?")

def _split_sentences_safe(t: str) -> list[str]:
    """Split sentences on .!? but avoid splitting decimal numbers (e.g., 2.5)."""
    if not t:
        return []
    parts = [s.strip() for s in re.split(r'(?<!\d)[\.!?]+(?!\d)', t) if s.strip()]
    return parts



def _count_numbers_in_sentence(s: str) -> int:
    if not s:
        return 0
    return len(_NUM_RE.findall(s))


def validate_dna_line(
    text: Any,
    banned_words: Optional[Sequence[str]] = None,
    forbid_tokens: Optional[Sequence[str]] = None,
    max_numbers_per_sentence: int = 2,
) -> Dict[str, Any]:
    """Validate DNALine against hard guardrails.

    Returns a dict:
      {"ok": bool, "issues": [str, ...]}

    This function is intentionally conservative: it flags early rather than
    attempting to repair text.
    """
    t = "" if text is None else str(text)
    t_strip = t.strip()
    issues: List[str] = []

    if not t_strip:
        return {"ok": False, "issues": ["EMPTY"]}

    tl = t_strip.lower()

    # banned words
    bw = [str(x).lower().strip() for x in (banned_words or []) if str(x).strip()]
    for w in bw:
        if w and w in tl:
            issues.append(f"BANNED_WORD:{w}")

    # forbidden MS/CV mentions
    ft = [str(x).lower().strip() for x in (forbid_tokens or []) if str(x).strip()]
    for tok in ft:
        if tok and tok in tl:
            issues.append(f"FORBID_TOKEN:{tok}")

    # number density guardrail (per sentence)
    # Split on ., !, ? while keeping it simple.
    sentences = _split_sentences_safe(t_strip)
    if not sentences:
        sentences = [t_strip]

    for i, s in enumerate(sentences, start=1):
        n = _count_numbers_in_sentence(s)
        if n > int(max_numbers_per_sentence):
            issues.append(f"TOO_MANY_NUMBERS:S{i}:{n}")

    return {"ok": len(issues) == 0, "issues": issues}


def validate_short_text(
    text: Any,
    *,
    max_sentences: int = 2,
    max_numbers_per_sentence: int = 2,
    forbid_bullets: bool = True,
) -> Dict[str, Any]:
    """Validate a short narrative line (Status/Plan) for rewrite-only.

    Guardrails (conservative):
    - Must not be empty.
    - Must be within max_sentences (split on . ! ?).
    - Each sentence must not exceed max_numbers_per_sentence.
    - Optionally forbid bullet-like markers and newlines.

    Returns {"ok": bool, "issues": [str, ...]}.
    """
    t = "" if text is None else str(text)
    t_strip = t.strip()
    issues: List[str] = []

    if not t_strip:
        return {"ok": False, "issues": ["EMPTY"]}

    if "\n" in t_strip or "\r" in t_strip:
        issues.append("NEWLINE")

    if forbid_bullets:
        # Common bullet tokens
        if re.search(r"(^|\s)([-*•]|\d+\))\s+", t_strip):
            issues.append("BULLET_STYLE")

    sentences = _split_sentences_safe(t_strip)
    if not sentences:
        sentences = [t_strip]

    if len(sentences) > int(max_sentences):
        issues.append(f"TOO_MANY_SENTENCES:{len(sentences)}")

    for i, s in enumerate(sentences, start=1):
        n = _count_numbers_in_sentence(s)
        if n > int(max_numbers_per_sentence):
            issues.append(f"TOO_MANY_NUMBERS:S{i}:{n}")

    return {"ok": len(issues) == 0, "issues": issues}


def validate_contains_phrases(text: Any, required_phrases: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Ensure all required phrases appear verbatim in text."""
    t = "" if text is None else str(text)
    t_strip = t.strip()
    req = [str(x) for x in (required_phrases or []) if str(x).strip()]
    issues: List[str] = []
    if not t_strip:
        return {"ok": False, "issues": ["EMPTY"]}
    for p in req:
        if p not in t_strip:
            issues.append(f"MISSING_PHRASE:{p}")
    return {"ok": len(issues) == 0, "issues": issues}
