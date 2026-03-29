"""
Provider Benchmark & Telemetry Layer.

Maintains a live registry of LLM provider profiles (OpenAI, Anthropic, Google,
Local) with rolling telemetry: average latency, cost-per-1K-tokens, quality
trend, error rate, and rate-limit counters.  The RL agent consumes these
profiles as static context for its routing decisions.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 50


@dataclass
class ProviderProfile:
    """Snapshot of a single provider's telemetry."""
    name: str
    avg_latency: float
    cost_per_1k_tokens: float
    quality_trend: float
    error_rate: float = 0.0
    rate_limit_hits: int = 0
    total_requests: int = 0
    last_rate_limit_ts: float = 0.0

    @property
    def is_rate_limited(self) -> bool:
        """True if a 429 was received in the last 5 seconds."""
        return (time.time() - self.last_rate_limit_ts) < 5.0


class ProviderRegistry:
    """
    Extensible registry that tracks real-time telemetry per provider.

    Providers are seeded with baseline stats and continuously updated as
    responses arrive.  Any new provider can be registered at runtime via
    ``register()``.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, ProviderProfile] = {}
        self._latency_window: Dict[str, List[float]] = {}
        self._quality_window: Dict[str, List[float]] = {}
        self._error_window: Dict[str, List[bool]] = {}
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        defaults = [
            ("OpenAI-GPT4", 0.75, 0.030, 0.93),
            ("Anthropic-Claude", 0.42, 0.015, 0.89),
            ("Google-Gemini", 0.35, 0.010, 0.84),
            ("Local-Llama3-8B", 0.24, 0.002, 0.64),
            ("Local-Phi3-Mini", 0.12, 0.001, 0.52),
        ]
        for name, lat, cost, qual in defaults:
            self.register(name, avg_latency=lat, cost_per_1k_tokens=cost, quality_trend=qual)

    def register(
        self,
        name: str,
        *,
        avg_latency: float = 0.5,
        cost_per_1k_tokens: float = 0.01,
        quality_trend: float = 0.80,
    ) -> None:
        self._profiles[name] = ProviderProfile(
            name=name,
            avg_latency=avg_latency,
            cost_per_1k_tokens=cost_per_1k_tokens,
            quality_trend=quality_trend,
        )
        self._latency_window[name] = []
        self._quality_window[name] = []
        self._error_window[name] = []
        logger.info("Registered provider '%s'", name)

    def get(self, name: str) -> Optional[ProviderProfile]:
        return self._profiles.get(name)

    def all_profiles(self) -> Dict[str, ProviderProfile]:
        return dict(self._profiles)

    @property
    def provider_names(self) -> List[str]:
        return list(self._profiles.keys())

    def record_response(
        self,
        name: str,
        latency: float,
        quality: float,
        success: bool,
        rate_limited: bool,
    ) -> None:
        """Ingest a provider response and refresh rolling stats."""
        prof = self._profiles.get(name)
        if prof is None:
            logger.warning("record_response for unknown provider '%s'", name)
            return

        prof.total_requests += 1

        win_lat = self._latency_window[name]
        win_lat.append(latency)
        if len(win_lat) > _ROLLING_WINDOW:
            win_lat.pop(0)
        prof.avg_latency = float(np.mean(win_lat))

        win_q = self._quality_window[name]
        win_q.append(quality)
        if len(win_q) > _ROLLING_WINDOW:
            win_q.pop(0)
        prof.quality_trend = float(np.mean(win_q))

        win_err = self._error_window[name]
        win_err.append(not success)
        if len(win_err) > _ROLLING_WINDOW:
            win_err.pop(0)
        prof.error_rate = float(np.mean(win_err))

        if rate_limited:
            prof.rate_limit_hits += 1
            prof.last_rate_limit_ts = time.time()
