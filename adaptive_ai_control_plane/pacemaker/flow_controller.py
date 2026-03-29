"""Pacemaker Flow Controller — heartbeat-based burst detection and traffic shaping."""

from __future__ import annotations

from collections import deque
import time


class FlowController:
    """Monitors the rate of incoming requests.  When more than
    ``burst_threshold`` prompts arrive within ``window_sec`` seconds the
    controller enters **Burst_Mode**, signalling the orchestrator to shed
    non-essential traffic to the cheapest local model so that VIP SLAs are
    protected."""

    def __init__(
        self,
        burst_threshold: int = 10,
        window_sec: float = 5.0,
        cooldown_sec: float = 10.0,
    ) -> None:
        self._burst_threshold = burst_threshold
        self._window_sec = window_sec
        self._cooldown_sec = cooldown_sec
        self._heartbeats: deque = deque()
        self._burst_activated_at: float | None = None
        self._total_beats: int = 0

    def heartbeat(self, ts: float | None = None) -> None:
        """Register one incoming request."""
        ts = ts if ts is not None else time.time()
        self._heartbeats.append(ts)
        self._total_beats += 1
        self._prune(ts)

    @property
    def mode(self) -> str:
        now = time.time()
        self._prune(now)

        if self._burst_activated_at is not None:
            if now - self._burst_activated_at < self._cooldown_sec:
                return "Burst_Mode"
            self._burst_activated_at = None

        if len(self._heartbeats) >= self._burst_threshold:
            self._burst_activated_at = now
            return "Burst_Mode"

        return "Normal"

    @property
    def is_burst(self) -> bool:
        return self.mode == "Burst_Mode"

    @property
    def load_index(self) -> int:
        """0 = Normal, 1 = Burst_Mode."""
        return 1 if self.is_burst else 0

    @property
    def current_rate(self) -> int:
        self._prune(time.time())
        return len(self._heartbeats)

    @property
    def total_processed(self) -> int:
        return self._total_beats

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_sec
        while self._heartbeats and self._heartbeats[0] < cutoff:
            self._heartbeats.popleft()
