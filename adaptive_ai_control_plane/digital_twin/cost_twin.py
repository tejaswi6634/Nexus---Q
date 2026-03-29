"""Cost Twin — forecasts whether the current request will breach the daily budget."""

from __future__ import annotations

from adaptive_ai_control_plane.digital_twin.simulation_engine import SimulationEngine


class CostTwin:
    """Tracks cumulative daily spend and predicts budget overruns before they
    happen, allowing the orchestrator to pre-emptively downgrade providers."""

    def __init__(self, daily_budget: float, engine: SimulationEngine) -> None:
        self._daily_budget = daily_budget
        self._engine = engine
        self._spent: float = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(self._daily_budget - self._spent, 0.0)

    @property
    def utilization_pct(self) -> float:
        if self._daily_budget == 0:
            return 100.0
        return (self._spent / self._daily_budget) * 100.0

    def record_cost(self, amount: float) -> None:
        self._spent += amount

    def would_exceed(self, estimated_cost: float) -> bool:
        return (self._spent + estimated_cost) > self._daily_budget

    def forecast_status(self, estimated_cost: float) -> str:
        if self.would_exceed(estimated_cost):
            return "Budget_Exceeded"
        if self.utilization_pct > 80:
            return "Budget_Warning"
        return "Budget_OK"

    def reset(self) -> None:
        self._spent = 0.0
