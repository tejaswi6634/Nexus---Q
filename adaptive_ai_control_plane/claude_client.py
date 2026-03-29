"""Anthropic Messages API — live inference for Claude when routed."""

from __future__ import annotations

import time
from typing import Tuple

from adaptive_ai_control_plane import settings


def _message_text(content: list) -> str:
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost from token usage (per-million rates; defaults to Sonnet-tier)."""
    m = model.lower()
    if "haiku" in m:
        inp_m, out_m = 0.8, 4.0
    elif "opus" in m:
        inp_m, out_m = 15.0, 75.0
    else:
        inp_m, out_m = 3.0, 15.0
    return (input_tokens * inp_m + output_tokens * out_m) / 1_000_000.0


def complete_claude(
    prompt: str,
    *,
    quality_hint: float = 0.92,
) -> Tuple[str, float, float, float]:
    """Call Claude Messages API.

    Returns ``(text, latency_sec, cost_usd, quality)``.
    """
    from anthropic import Anthropic

    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.ANTHROPIC_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    text = _message_text(msg.content)
    usage = msg.usage
    inp = getattr(usage, "input_tokens", 0)
    out = getattr(usage, "output_tokens", 0)
    cost = _estimate_cost_usd(settings.ANTHROPIC_MODEL, inp, out)
    quality = min(0.99, max(quality_hint, 0.88))
    return text, latency, cost, quality
