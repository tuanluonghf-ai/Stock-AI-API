PHASE H (H.1 + H.2) — Narrative Regression & Release Governance

What’s included:
- ui/narrative_governance.py: key ordering + sentence budget + validation
- ui/render_appendix.py patched: NARRATIVE (KEYED) uses validate_narrative() and shows guardrail warnings in an expander.
- tools/generate_golden_narrative.py: generate golden snapshots into golden/narrative/
- tools/check_golden_narrative.py: regression check vs golden snapshots

How to use:
1) Copy the 'ui' folder into your repo at: inception/ui/ (overwrite same-name files).
2) Copy 'tools' and 'golden' folders to repo root (next to app.py).
3) Restart Streamlit.

Generate golden (one-time):
  python tools/generate_golden_narrative.py --tickers VCB CII MSN --data-dir . --price-vol Price_Vol.xlsx --position-mode FLAT

Check regression (CI / before release):
  python tools/check_golden_narrative.py --tickers VCB CII MSN --data-dir . --price-vol Price_Vol.xlsx --position-mode FLAT
