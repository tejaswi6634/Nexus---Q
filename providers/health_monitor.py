"""
Dynamic Endpoint Health Scorer.

Computes a composite health score for every registered provider endpoint
using the weighted formula:

    Score = w1 * norm(latency) + w2 * error_rate + w3 * queue_load

Lower scores indicate healthier endpoints.  The ``HealthMonitor`` acts as a
primary pre-filter: the router should prefer providers with the lowest score.
All heavy-path arithmetic is numpy-vectorised for sub-millisecond overhead.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .registry import ProviderRegistry

logger = logging.getLogger(__name__)

_DEFAULT_W1 = 0.45   # latency weight
_DEFAULT_W2 = 0.35   # error-rate weight
_DEFAULT_W3 = 0.20   # queue-load weight

_LATENCY_CAP = 2.0   # seconds -- used to normalise latency into [0, 1]


class HealthMonitor:
    """
    Real-time health scorer for every provider in the ``ProviderRegistry``.

    Parameters
    ----------
    registry : ProviderRegistry
        Live reference; the monitor reads the latest telemetry on each call.
    w1, w2, w3 : float
        Weights for latency, error_rate, and queue_load respectively.
    degraded_threshold : float
        Providers whose score exceeds this value are flagged as *degraded*
        and should be deprioritised by the router.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        w1: float = _DEFAULT_W1,
        w2: float = _DEFAULT_W2,
        w3: float = _DEFAULT_W3,
        degraded_threshold: float = 0.65,
    ) -> None:
        self.registry = registry
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.degraded_threshold = degraded_threshold
        self._queue_loads: Dict[str, float] = {}

    def update_queue_load(self, provider: str, queue_fraction: float) -> None:
        """Set the current queue-utilisation fraction ∈ [0, 1] for *provider*."""
        self._queue_loads[provider] = np.clip(queue_fraction, 0.0, 1.0)

    def score(self, provider: str) -> float:
        """
        Compute the composite health score (lower is better) for *provider*.
        """
        prof = self.registry.get(provider)
        if prof is None:
            return 1.0  # unknown → worst possible

        norm_latency = np.clip(prof.avg_latency / _LATENCY_CAP, 0.0, 1.0)
        error_rate = np.clip(prof.error_rate, 0.0, 1.0)
        queue_load = self._queue_loads.get(provider, 0.0)

        return float(
            self.w1 * norm_latency
            + self.w2 * error_rate
            + self.w3 * queue_load
        )

    def score_all(self) -> Dict[str, float]:
        """Return ``{provider_name: health_score}`` for every registered provider."""
        return {name: self.score(name) for name in self.registry.provider_names}

    def best_provider(self) -> Optional[str]:
        """Return the provider name with the lowest (healthiest) score."""
        scores = self.score_all()
        if not scores:
            return None
        return min(scores, key=scores.get)  # type: ignore[arg-type]

    def is_degraded(self, provider: str) -> bool:
        return self.score(provider) > self.degraded_threshold

    def healthy_providers(self) -> Dict[str, float]:
        """Return only non-degraded providers with their scores."""
        return {
            name: sc
            for name, sc in self.score_all().items()
            if sc <= self.degraded_threshold
        }
