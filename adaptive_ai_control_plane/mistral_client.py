"""Mistral Chat Completions — live inference when Mistral-7B is routed.

Uses the OpenAI-compatible API at ``https://api.mistral.ai/v1`` (see Mistral docs).
"""

from __future__ import annotations

import time
from typing import Tuple

from adaptive_ai_control_plane import settings


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost from token usage (per-million; model-dependent defaults)."""
    m = model.lower()
    if "mistral-large" in m:
        inp_m, out_m = 2.0, 6.0
    elif "mistral-small" in m or "open-mistral" in m or "mistral-7b" in m:
        inp_m, out_m = 0.20, 0.20
    else:
        inp_m, out_m = 0.20, 0.20
    return (input_tokens * inp_m + output_tokens * out_m) / 1_000_000.0


def complete_mistral(
    prompt: str,
    *,
    quality_hint: float = 0.75,
) -> Tuple[str, float, float, float]:
    """Call Mistral chat API.

    Returns ``(text, latency_sec, cost_usd, quality)``.
    """
    from openai import OpenAI

    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set")

    client = OpenAI(
        api_key=settings.MISTRAL_API_KEY,
        base_url=settings.MISTRAL_API_BASE,
    )
    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=settings.MISTRAL_MODEL,
        max_tokens=settings.MISTRAL_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    text = (completion.choices[0].message.content or "").strip()
    usage = completion.usage
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    cost = _estimate_cost_usd(settings.MISTRAL_MODEL, inp, out)
    quality = min(0.99, max(quality_hint, 0.72))
    return text, latency, cost, quality
