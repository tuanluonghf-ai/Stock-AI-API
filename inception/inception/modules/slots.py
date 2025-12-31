"""Reserved module slots for future expansion.

These are intentionally thin placeholders to keep the architecture stable.
When you implement a new module (portfolio/news/fundamentals), add a new
file in inception/modules/ and call register('<name>', run) with the runner
signature: run(analysis_pack: dict, ctx: dict) -> ModuleResult.

Recommended canonical names:
- portfolio
- fundamental_summary
- news
"""
