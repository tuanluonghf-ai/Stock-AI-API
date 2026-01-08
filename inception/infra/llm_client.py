"""LLM client adapter (OpenAI) for INCEPTION.

Design goals:
- Keep Streamlit optional (secrets fallback).
- Keep app.py thin: all API calling lives here.
- Provide a single entrypoint for chat completions.

This module intentionally does not depend on any UI renderer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import os


def get_openai_api_key() -> Optional[str]:
    """Resolve OPENAI_API_KEY.

    Priority:
      1) env var OPENAI_API_KEY
      2) Streamlit secrets OPENAI_API_KEY (if available)

    If found via secrets, it will also be copied into os.environ
    so downstream libraries that read env work as expected.
    """
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key

    # Optional Streamlit secrets
    try:
        import streamlit as st  # type: ignore

        sec = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        if sec:
            key = str(sec).strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
                return key
    except Exception:
        pass

    return None


def call_openai_chat(
    *,
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1600,
    system: str = "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư.",
) -> str:
    """Call OpenAI chat completions.

    Raises:
      RuntimeError if API key is missing or the OpenAI SDK is unavailable.
    """
    key = get_openai_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"openai package not available: {e}") from e

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
