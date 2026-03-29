"""Load ``.env`` from the ``hybrid_qc_framework`` root and expose settings."""

from __future__ import annotations

import os
from pathlib import Path

_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_FRAMEWORK_ROOT / ".env")


load_env()


def _truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


OPENAI_API_KEY: str = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL: str = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()


def _openai_max_tokens() -> int:
    raw = os.environ.get("OPENAI_MAX_TOKENS", "1024")
    try:
        return max(64, min(8192, int(raw)))
    except ValueError:
        return 1024


OPENAI_MAX_TOKENS: int = _openai_max_tokens()
# Deprecated: ignored. Live OpenAI is only used when the user supplies prompt(s), not for corpus simulation.
OPENAI_LIVE_ALWAYS: bool = _truthy(os.environ.get("OPENAI_LIVE_ALWAYS"))

GOOGLE_API_KEY: str = (os.environ.get("GOOGLE_API_KEY") or "").strip()
GOOGLE_MODEL: str = (os.environ.get("GOOGLE_MODEL") or "gemini-2.0-flash").strip()


def _google_max_tokens() -> int:
    raw = os.environ.get("GOOGLE_MAX_TOKENS", "1024")
    try:
        return max(64, min(8192, int(raw)))
    except ValueError:
        return 1024


GOOGLE_MAX_TOKENS: int = _google_max_tokens()
# Deprecated: ignored. Live Gemini is only used when the user supplies prompt(s).
GOOGLE_LIVE_ALWAYS: bool = _truthy(os.environ.get("GOOGLE_LIVE_ALWAYS"))

ANTHROPIC_API_KEY: str = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
ANTHROPIC_MODEL: str = (
    os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
).strip()


def _max_tokens() -> int:
    raw = os.environ.get("ANTHROPIC_MAX_TOKENS", "1024")
    try:
        return max(64, min(8192, int(raw)))
    except ValueError:
        return 1024


ANTHROPIC_MAX_TOKENS: int = _max_tokens()
# Deprecated: ignored. Live Claude is only used when the user supplies prompt(s).
ANTHROPIC_LIVE_ALWAYS: bool = _truthy(os.environ.get("ANTHROPIC_LIVE_ALWAYS"))


def should_use_live_claude(user_supplied_prompts: bool) -> bool:
    """Call Claude API only when the user supplied prompt(s). Corpus simulation never calls the API."""
    return bool(ANTHROPIC_API_KEY) and user_supplied_prompts


def should_use_live_openai(user_supplied_prompts: bool) -> bool:
    """Call OpenAI only when the user supplied prompt(s). Corpus simulation never calls the API."""
    return bool(OPENAI_API_KEY) and user_supplied_prompts


def should_use_live_google(user_supplied_prompts: bool) -> bool:
    """Call Gemini only when the user supplied prompt(s). Corpus simulation never calls the API."""
    return bool(GOOGLE_API_KEY) and user_supplied_prompts


MISTRAL_API_KEY: str = (os.environ.get("MISTRAL_API_KEY") or "").strip()
MISTRAL_MODEL: str = (os.environ.get("MISTRAL_MODEL") or "mistral-small-latest").strip()
MISTRAL_API_BASE: str = (
    os.environ.get("MISTRAL_API_BASE") or "https://api.mistral.ai/v1"
).strip().rstrip("/")
if not MISTRAL_API_BASE.endswith("/v1"):
    MISTRAL_API_BASE = f"{MISTRAL_API_BASE}/v1"


def _mistral_max_tokens() -> int:
    raw = os.environ.get("MISTRAL_MAX_TOKENS", "1024")
    try:
        return max(64, min(8192, int(raw)))
    except ValueError:
        return 1024


MISTRAL_MAX_TOKENS: int = _mistral_max_tokens()


def should_use_live_mistral(user_supplied_prompts: bool) -> bool:
    """Call Mistral only when the user supplied prompt(s). Corpus simulation never calls the API."""
    return bool(MISTRAL_API_KEY) and user_supplied_prompts
