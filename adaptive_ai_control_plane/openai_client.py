"""OpenAI Chat Completions — live inference when GPT-4 is routed."""

from __future__ import annotations

import time
from typing import Tuple

from adaptive_ai_control_plane import settings


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost from token usage (per-million; defaults to gpt-4o-mini–tier)."""
    m = model.lower()
    if "gpt-4o" in m and "mini" not in m:
        inp_m, out_m = 2.5, 10.0
    elif "gpt-4o-mini" in m or ("gpt-4o" in m and "mini" in m):
        inp_m, out_m = 0.15, 0.60
    elif "gpt-3.5" in m:
        inp_m, out_m = 0.50, 1.50
    else:
        inp_m, out_m = 0.15, 0.60
    return (input_tokens * inp_m + output_tokens * out_m) / 1_000_000.0


def complete_openai(
    prompt: str,
    *,
    quality_hint: float = 0.95,
) -> Tuple[str, float, float, float]:
    """Call OpenAI Chat Completions API.

    Returns ``(text, latency_sec, cost_usd, quality)``.
    """
    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    text = (completion.choices[0].message.content or "").strip()
    usage = completion.usage
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    cost = _estimate_cost_usd(settings.OPENAI_MODEL, inp, out)
    quality = min(0.99, max(quality_hint, 0.88))
    return text, latency, cost, quality
