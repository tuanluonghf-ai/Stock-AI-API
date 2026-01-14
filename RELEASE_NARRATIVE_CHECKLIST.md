# INCEPTION — Narrative Release Checklist (H.3)

This checklist is **UI-side governance** for *Keyed Narrative* (Appendix E).

## 1) Mandatory pre-release commands

Generate (only when you intentionally update baseline):
```powershell
python tools/generate_golden_narrative.py --tickers VCB CII MSN --data-dir . --price-vol Price_Vol.xlsx --position-mode FLAT --timeframe D
```

Check (run before every release):
```powershell
python tools/check_golden_narrative.py --tickers VCB CII MSN --data-dir . --price-vol Price_Vol.xlsx --position-mode FLAT --timeframe D --strict
```

Expected:
- `[OK]` for all tickers
- exit code `0`

## 2) Drift policy

- **Unintentional drift**: treat as regression → fix or rollback.
- **Intentional drift**: regenerate golden *once* and commit the updated `golden/narrative/*.json`.

## 3) Guardrails enforced in strict mode

- Key order: `DNA → ZONE → BIAS → SIZE → DECISION → SAFETY`
- Max sentences: from `render_policy.max_sentences` (default 5)
- Must include `SAFETY_LINE`
- Numbers per sentence: <= 2 (soft but CI-enforced)

## 4) CI gate recommendation

Use the strict check as a CI step. Failure should block merge/release.

## 5) Versioning note

When you change phrase bank / resolver / key mapping:
- bump app version (e.g., 16.1 → 16.2)
- run `check_golden... --strict`
- only then release
