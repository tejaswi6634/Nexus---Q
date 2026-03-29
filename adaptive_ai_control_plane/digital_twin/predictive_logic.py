"""Predictive Logic — detects latency degradation trends across providers."""

from __future__ import annotations

from typing import Dict

from adaptive_ai_control_plane.digital_twin.simulation_engine import SimulationEngine


class PredictiveLogic:
    """Flags a provider as *Predicted_Degradation* if its average latency over
    the last ``recent_window`` calls has increased by more than ``threshold``
    compared to the preceding window of the same size."""

    THRESHOLD = 0.15  # 15 %
    RECENT_WINDOW = 5

    def __init__(self, engine: SimulationEngine) -> None:
        self._engine = engine

    def health_status(self, provider: str) -> str:
        series = self._engine.latency_series(provider)
        if len(series) < self.RECENT_WINDOW * 2:
            return "Healthy"

        recent = series[-self.RECENT_WINDOW:]
        preceding = series[-self.RECENT_WINDOW * 2: -self.RECENT_WINDOW]

        avg_recent = sum(recent) / len(recent)
        avg_preceding = sum(preceding) / len(preceding)

        if avg_preceding == 0:
            return "Healthy"

        increase_pct = (avg_recent - avg_preceding) / avg_preceding
        return "Predicted_Degradation" if increase_pct > self.THRESHOLD else "Healthy"

    def all_statuses(self, providers: list[str]) -> Dict[str, str]:
        return {p: self.health_status(p) for p in providers}

    def health_index(self, provider: str) -> int:
        """0 = Healthy, 1 = Predicted_Degradation."""
        return 0 if self.health_status(provider) == "Healthy" else 1
