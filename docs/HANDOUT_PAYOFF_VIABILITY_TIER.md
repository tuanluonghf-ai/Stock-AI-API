# Payoff Viability Tier v1 + Step 3 (Spec-Lock)

## A) Context
Payoff is an independent economic axis that describes how much potential swing is available relative to price; it is not the same as risk, volatility, or trend, and must be treated as a separate constraint.

## B) Definitions
- `payoff_span_norm = (max_swing_high - min_swing_low) / reference_price`
- Tier thresholds:
  - LOW < 0.15
  - MEDIUM < 0.40
  - HIGH >= 0.40
- Pack location in analysis_pack: `analysis_pack["payoff"]`

## C) Placement in pipeline
```
Stock DNA -> PayoffTier -> Class/Style -> Investor Fit -> Persona Fit -> Narrative
```

## D) Step 3 rules (minimal, as implemented)
- Persona cap:
  - LOW blocks Speculator/Compounder -> fallback CashFlowTrader
  - MEDIUM blocks Compounder -> fallback CashFlowTrader
- Narrative anchor:
  - If LOW/MEDIUM and persona in {Speculator, AlphaHunter, Compounder} and narrative lacks trade-off cues,
    append one Vietnamese sentence about payoff trade-off/opportunity cost.

## E) Governance / Non-goals
- No UI changes, no decision changes, no promise language, deterministic outputs.

## F) Validation checklist
- sanity_gate PASS
- regression_harness PASS
- challenge audit: mismatch=0, narrative_missing=0
