"""
Multi-Level Local Fallback Hierarchy — the 'Survival' Layer.

When all cloud providers are unavailable (circuit-breakers open, health
degraded, rate-limited) the hierarchy cascades through progressively
lighter local resources:

    Level 1 — ``Local-Llama-3-8B``  (full local model via Ollama / vLLM)
    Level 2 — ``Local-Phi-3-Mini``  (distilled model, optimised for speed)
    Level 3 — ``SemanticCache``     (closest historical match from the
                                     vector-similarity cache)

Each level is attempted in order; the first successful result is
returned.  If all three levels fail the request is marked as a total
system failure and an appropriate error response is synthesised.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Dict, Optional, Tuple

from .semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

_LOCAL_PROFILES: Dict[str, Dict[str, object]] = {
    "Local-Llama3-8B": {
        "latency": 0.24,
        "cost_mult": 0.15,
        "quality": 0.64,
        "template": (
            "Local Llama-3-8B response with solid general knowledge "
            "and reasonable detail covering the key aspects of the request."
        ),
    },
    "Local-Phi3-Mini": {
        "latency": 0.12,
        "cost_mult": 0.08,
        "quality": 0.52,
        "template": (
            "Local Phi-3-Mini distilled response covering essential "
            "points concisely with adequate accuracy."
        ),
    },
}

_CASCADE_ORDER = ["Local-Llama3-8B", "Local-Phi3-Mini"]


class LocalFallbackHierarchy:
    """
    Cascading local fallback manager.

    Parameters
    ----------
    semantic_cache : SemanticCache
        Level 3 fallback — used when both local models are unavailable.
    provider_call_fn : callable, optional
        Function ``(provider_name, estimated_cost) -> result_dict``.
        Defaults to a built-in simulation function.
    """

    def __init__(
        self,
        semantic_cache: SemanticCache,
        provider_call_fn: Optional[Callable] = None,
    ) -> None:
        self._cache = semantic_cache
        self._call_fn = provider_call_fn or self._simulated_call
        self._availability: Dict[str, bool] = {p: True for p in _CASCADE_ORDER}

    def set_availability(self, provider: str, available: bool) -> None:
        self._availability[provider] = available

    # ── Built-in simulation (used when no real Ollama/vLLM is wired) ─

    @staticmethod
    def _simulated_call(provider: str, estimated_cost: float) -> dict:
        prof = _LOCAL_PROFILES.get(provider, _LOCAL_PROFILES["Local-Phi3-Mini"])
        jitter = random.uniform(-0.03, 0.05)
        return {
            "provider": provider,
            "latency": max(0.02, float(prof["latency"]) + jitter),
            "cost": max(1e-6, estimated_cost * float(prof["cost_mult"])),
            "quality": max(0.0, min(1.0, float(prof["quality"]) + random.uniform(-0.05, 0.03))),
            "success": True,
            "rate_limited": False,
            "response_text": str(prof["template"]),
        }

    # ── Cascade Execution ────────────────────────────────────────────

    def execute(
        self, prompt: str, estimated_cost: float,
    ) -> Tuple[Optional[dict], str]:
        """
        Try each fallback level in order.

        Returns
        -------
        tuple[dict | None, str]
            ``(result_dict, level_tag)`` on success, or
            ``(None, "all_failed")`` if every level is exhausted.
        """
        # Level 1: Local-Llama3-8B
        if self._availability.get("Local-Llama3-8B", False):
            try:
                result = self._call_fn("Local-Llama3-8B", estimated_cost)
                if result.get("success"):
                    logger.info("Survival cascade: Level 1 (Llama-3-8B) ✓")
                    return result, "Level-1:Local-Llama3-8B"
            except Exception as exc:
                logger.warning("Survival cascade: Level 1 failed — %s", exc)

        # Level 2: Local-Phi3-Mini
        if self._availability.get("Local-Phi3-Mini", False):
            try:
                result = self._call_fn("Local-Phi3-Mini", estimated_cost)
                if result.get("success"):
                    logger.info("Survival cascade: Level 2 (Phi-3-Mini) ✓")
                    return result, "Level-2:Local-Phi3-Mini"
            except Exception as exc:
                logger.warning("Survival cascade: Level 2 failed — %s", exc)

        # Level 3: Semantic Cache
        cache_hit = self._cache.lookup(prompt)
        if cache_hit is not None:
            response_text, similarity, orig_provider = cache_hit
            logger.info(
                "Survival cascade: Level 3 (SemanticCache) ✓  sim=%.3f", similarity,
            )
            return {
                "provider": f"SemanticCache(orig={orig_provider})",
                "latency": 0.005,
                "cost": 0.0,
                "quality": similarity * 0.70,
                "success": True,
                "rate_limited": False,
                "response_text": response_text,
            }, "Level-3:SemanticCache"

        logger.error("Survival cascade: ALL levels exhausted")
        return None, "all_failed"
