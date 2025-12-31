"""INCEPTION Core Types

This file defines lightweight schemas for canonical shared packs.
Keep this intentionally minimal; expand fields as needed.

Design rules:
- Packs must be JSON-serializable (scalar/list/dict; no pandas objects).
- AnalysisPack is read-only for downstream modules (do not mutate).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


JSONValue = Any  # pragmatic; enforced via runtime validation (see modules.base)


class PrimarySetup(TypedDict, total=False):
    Name: str
    RiskPct: float
    RewardPct: float
    RR: float
    Probability: str


class MasterScorePack(TypedDict, total=False):
    Total: float
    Breakdown: Dict[str, float]


class Scenario12Pack(TypedDict, total=False):
    Name: str
    Code: str
    Notes: str


class AnalysisPack(TypedDict, total=False):
    _schema_version: str
    PriceContext: Dict[str, JSONValue]
    RSIContext: Dict[str, JSONValue]
    VolumeContext: Dict[str, JSONValue]
    LevelContext: Dict[str, JSONValue]
    Market: Dict[str, JSONValue]
    PrimarySetup: PrimarySetup
    MasterScore: MasterScorePack
    Scenario12: Scenario12Pack


class CharacterPack(TypedDict, total=False):
    CharacterClass: str
    Flags: List[str]
    CoreStats: Dict[str, float]
    CombatStats: Dict[str, float]
    Conviction: Dict[str, float]
    Error: str
    _Ticker: str


class ModuleResult(TypedDict, total=False):
    ok: bool
    name: str
    payload: Dict[str, JSONValue]
    error: Optional[str]
    meta: Dict[str, JSONValue]
