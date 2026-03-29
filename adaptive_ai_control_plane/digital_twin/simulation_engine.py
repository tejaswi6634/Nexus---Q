"""Digital Twin Simulation Engine — tracks the last N responses per provider."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ResponseRecord:
    latency: float
    cost: float
    quality: float
    timestamp: float


class SimulationEngine:
    """Maintains a rolling window of the most recent responses for every
    provider so that downstream predictive components have a live data feed."""

    WINDOW_SIZE = 20

    def __init__(self) -> None:
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.WINDOW_SIZE)
        )

    def record(
        self, provider: str, latency: float, cost: float, quality: float, ts: float
    ) -> None:
        self._history[provider].append(
            ResponseRecord(latency=latency, cost=cost, quality=quality, timestamp=ts)
        )

    def recent(self, provider: str, n: Optional[int] = None) -> List[ResponseRecord]:
        history = list(self._history[provider])
        return history[-n:] if n else history

    def latency_series(self, provider: str) -> List[float]:
        return [r.latency for r in self._history[provider]]

    def cost_series(self, provider: str) -> List[float]:
        return [r.cost for r in self._history[provider]]

    def has_data(self, provider: str) -> bool:
        return len(self._history[provider]) > 0

    def total_records(self, provider: str) -> int:
        return len(self._history[provider])
