"""Warden agent package (U3, ported from the warden build).

The agent does deterministic orchestration over the src engine and uses the LLM only for
narration (the LLM never computes a physical quantity). This module loads the repo .env so
ANTHROPIC_API_KEY is available whether the app is started from the repo root or elsewhere, and
resolves the narration/orchestration model.

Model default is Sonnet 4.6, matching the battery narrator (src/config.py) so the whole app reports
one model. Override with WARDEN_LLM_MODEL (e.g. claude-haiku-4-5 for cheaper/faster narration).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv

    _root = Path(__file__).resolve().parent.parent.parent  # src/agent -> src -> repo root
    _candidate = _root / ".env"
    if _candidate.is_file():
        load_dotenv(_candidate, override=False)
    _found = find_dotenv(usecwd=True)
    if _found:
        load_dotenv(_found, override=False)
except Exception:
    # dotenv is optional; an exported ANTHROPIC_API_KEY still works without it.
    pass


DEFAULT_LLM_MODEL = "claude-sonnet-4-6"  # matches src/config.py narration_model (one model app-wide)


def anthropic_key() -> str:
    """The Anthropic API key if configured, else empty string (callers fall back to template)."""
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def llm_model() -> str:
    """Narration/orchestration model id. Override with WARDEN_LLM_MODEL (e.g. claude-sonnet-4-6)."""
    return os.environ.get("WARDEN_LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL
