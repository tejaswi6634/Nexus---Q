"""
Predictive Health Engine — the 'Pre-Emptive' Layer.

Tracks a *time-bounded* rolling window of per-provider telemetry events
(latency, errors, timeouts) and computes real-time health metrics.

Degradation rules (configurable):
    * ``latency_moving_avg`` increases by > 50 % over baseline  →  DEGRADED
    * ``error_rate`` exceeds 5 % in the observation window      →  DEGRADED
    * ``error_rate`` > 30 % or ``timeout_freq`` > 25 %          →  DOWN

The router consults this engine **before** dispatching, allowing it to
steer traffic away from providers that are *about to fail* rather than
waiting for 4xx/5xx responses.

Thread-safe; can optionally run a background monitoring loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .models import HealthMetrics, ProviderStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _HealthEvent:
    timestamp: float
    latency: float
    is_error: bool
    is_timeout: bool


class ProviderHealthMonitor:
    """
    Maintains a rolling time-window of health events per provider and
    derives predictive health status from latency trends and error rates.

    Parameters
    ----------
    window_seconds : float
        Observation window length.  Only events within this window
        contribute to metric computation.
    latency_spike_threshold : float
        Fractional increase in ``latency_moving_avg`` over baseline
        that triggers a DEGRADED classification (0.50 = 50 %).
    error_rate_threshold : float
        Error rate above which a provider is classified DEGRADED.
    timeout_threshold : float
        Timeout frequency above which a provider is classified DOWN.
    min_samples : int
        Minimum events required before the engine changes status from
        the initial HEALTHY default.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        latency_spike_threshold: float = 0.50,
        error_rate_threshold: float = 0.05,
        timeout_threshold: float = 0.10,
        min_samples: int = 5,
    ) -> None:
        self._window_seconds = window_seconds
        self._latency_spike_threshold = latency_spike_threshold
        self._error_rate_threshold = error_rate_threshold
        self._timeout_threshold = timeout_threshold
        self._min_samples = min_samples

        self._events: Dict[str, Deque[_HealthEvent]] = {}
        self._baselines: Dict[str, float] = {}
        self._status_cache: Dict[str, ProviderStatus] = {}
        self._lock = threading.Lock()

        self._bg_thread: Optional[threading.Thread] = None
        self._running = False

    # ── Registration ─────────────────────────────────────────────────

    def register_provider(self, provider: str, baseline_latency: float) -> None:
        with self._lock:
            self._events.setdefault(provider, deque())
            self._baselines[provider] = baseline_latency
            self._status_cache[provider] = ProviderStatus.HEALTHY

    # ── Event Ingestion ──────────────────────────────────────────────

    def record_event(
        self,
        provider: str,
        latency: float,
        is_error: bool = False,
        is_timeout: bool = False,
    ) -> None:
        with self._lock:
            if provider not in self._events:
                self._events[provider] = deque()
                self._baselines.setdefault(provider, latency)
            self._events[provider].append(
                _HealthEvent(time.time(), latency, is_error, is_timeout)
            )

    # ── Metric Computation ───────────────────────────────────────────

    def _prune(self, provider: str) -> None:
        cutoff = time.time() - self._window_seconds
        dq = self._events.get(provider)
        if dq:
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()

    def compute_metrics(self, provider: str) -> HealthMetrics:
        with self._lock:
            self._prune(provider)
            events = list(self._events.get(provider, []))

        n = len(events)
        if n == 0:
            return HealthMetrics(
                provider=provider,
                status=self._status_cache.get(provider, ProviderStatus.HEALTHY),
            )

        lats = [e.latency for e in events]
        lat_avg = sum(lats) / n
        baseline = self._baselines.get(provider, lat_avg)
        trend_pct = (lat_avg - baseline) / baseline if baseline > 0 else 0.0

        err_count = sum(1 for e in events if e.is_error)
        tmo_count = sum(1 for e in events if e.is_timeout)
        error_rate = err_count / n
        timeout_freq = tmo_count / n

        # Status classification
        if n >= self._min_samples:
            if error_rate > 0.30 or timeout_freq > 0.25:
                status = ProviderStatus.DOWN
            elif (
                trend_pct > self._latency_spike_threshold
                or error_rate > self._error_rate_threshold
            ):
                status = ProviderStatus.DEGRADED
            else:
                status = ProviderStatus.HEALTHY
        else:
            status = self._status_cache.get(provider, ProviderStatus.HEALTHY)

        with self._lock:
            self._status_cache[provider] = status
            if status == ProviderStatus.HEALTHY and n >= self._min_samples:
                self._baselines[provider] = baseline * 0.95 + lat_avg * 0.05

        return HealthMetrics(
            provider=provider,
            latency_moving_avg=lat_avg,
            latency_baseline=baseline,
            latency_trend_pct=trend_pct,
            error_rate=error_rate,
            timeout_frequency=timeout_freq,
            sample_count=n,
            status=status,
        )

    # ── Convenience Queries ──────────────────────────────────────────

    def get_status(self, provider: str) -> ProviderStatus:
        return self.compute_metrics(provider).status

    def get_all_metrics(self) -> Dict[str, HealthMetrics]:
        with self._lock:
            providers = list(self._events.keys())
        return {p: self.compute_metrics(p) for p in providers}

    def get_healthy_providers(self) -> List[str]:
        return [
            p for p, m in self.get_all_metrics().items()
            if m.status == ProviderStatus.HEALTHY
        ]

    def get_degraded_providers(self) -> List[str]:
        return [
            p for p, m in self.get_all_metrics().items()
            if m.status != ProviderStatus.HEALTHY
        ]

    # ── Background Monitoring ────────────────────────────────────────

    def start_background_monitor(self, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True
        self._bg_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True,
        )
        self._bg_thread.start()
        logger.info(
            "ProviderHealthMonitor background thread started (%.1fs interval)", interval
        )

    def stop_background_monitor(self) -> None:
        self._running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=3)

    def _monitor_loop(self, interval: float) -> None:
        while self._running:
            for p, m in self.get_all_metrics().items():
                if m.status == ProviderStatus.DEGRADED:
                    logger.warning(
                        "PREDICTIVE ALERT  %s → DEGRADED  "
                        "(latency +%.0f%%, error_rate %.1f%%)",
                        p, m.latency_trend_pct * 100, m.error_rate * 100,
                    )
                elif m.status == ProviderStatus.DOWN:
                    logger.error("PREDICTIVE ALERT  %s → DOWN", p)
            time.sleep(interval)

    # ── Async Façade ─────────────────────────────────────────────────

    async def async_compute_metrics(self, provider: str) -> HealthMetrics:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.compute_metrics, provider)
