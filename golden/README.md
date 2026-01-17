# Golden Regression Baseline

This folder stores *compact* golden snapshots used to detect:
- Schema drift (missing keys / new keys / type changes)
- Drift in high-signal outputs: TradePlan / Decision / NarrativeDraftPack

## Create baselines

```bash
python regression_golden.py --update --tickers VNM CII MSN
```

Expected files:
- golden/VNM.flat.json
- golden/CII.flat.json
- golden/MSN.flat.json
- golden/_manifest.json

## Contract semantics

The golden manifest includes a `packs.InvestorMappingPack` block that locks v1
semantics (scale, labels, required sections). Validation is enforced at the
DashboardSummary stage; earlier stages only validate when present to avoid
false failures.

## Compare against baselines

```bash
python regression_golden.py --tickers VNM CII MSN
```

## Schema-only compare (ignore values)

```bash
python regression_golden.py --schema-only --tickers VNM CII MSN
```
