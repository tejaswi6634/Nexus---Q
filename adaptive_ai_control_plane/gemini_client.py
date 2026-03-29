"""Google Generative AI (Gemini) — live inference when Gemini is routed."""

from __future__ import annotations

import time
from typing import Tuple

from adaptive_ai_control_plane import settings


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost from token usage (per-million; flash-tier defaults)."""
    m = model.lower()
    if "pro" in m:
        inp_m, out_m = 1.25, 5.0
    else:
        inp_m, out_m = 0.075, 0.30
    return (input_tokens * inp_m + output_tokens * out_m) / 1_000_000.0


def complete_gemini(
    prompt: str,
    *,
    quality_hint: float = 0.84,
) -> Tuple[str, float, float, float]:
    """Call Gemini generateContent.

    Returns ``(text, latency_sec, cost_usd, quality)``.
    """
    import google.generativeai as genai

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set")

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel(settings.GOOGLE_MODEL)
    t0 = time.perf_counter()
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": settings.GOOGLE_MAX_TOKENS},
    )
    latency = time.perf_counter() - t0
    try:
        text = (resp.text or "").strip()
    except ValueError:
        # Blocked or empty candidate (safety / no text parts)
        text = ""
    inp = out = 0
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        inp = int(getattr(um, "prompt_token_count", 0) or 0)
        out = int(getattr(um, "candidates_token_count", 0) or 0)
    cost = _estimate_cost_usd(settings.GOOGLE_MODEL, inp, out)
    quality = min(0.99, max(quality_hint, 0.82))
    return text, latency, cost, quality
