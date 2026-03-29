"""Savings Tracker — calculates Cost Avoided relative to GPT-4 baseline."""

from __future__ import annotations

from typing import List


class SavingsTracker:
    """Tracks the running total of *Cost Avoided* — the difference between what
    GPT-4 would have charged and what was actually spent.  Provides per-request
    and aggregate summaries for the finance dashboard."""

    def __init__(self) -> None:
        self._records: List[dict] = []
        self._total_actual: float = 0.0
        self._total_gpt4_equiv: float = 0.0

    def record(self, actual_cost: float, gpt4_equivalent_cost: float) -> float:
        saved = gpt4_equivalent_cost - actual_cost
        self._records.append({
            "actual": actual_cost,
            "gpt4_equiv": gpt4_equivalent_cost,
            "saved": saved,
        })
        self._total_actual += actual_cost
        self._total_gpt4_equiv += gpt4_equivalent_cost
        return saved

    @property
    def total_saved(self) -> float:
        return self._total_gpt4_equiv - self._total_actual

    @property
    def total_spent(self) -> float:
        return self._total_actual

    @property
    def total_gpt4_equivalent(self) -> float:
        return self._total_gpt4_equiv

    @property
    def savings_pct(self) -> float:
        if self._total_gpt4_equiv == 0:
            return 0.0
        return (self.total_saved / self._total_gpt4_equiv) * 100.0

    @property
    def request_count(self) -> int:
        return len(self._records)

    def summary(self) -> str:
        return (
            f"[Finance] {self.request_count} requests | "
            f"Spent: ${self.total_spent:.4f} | "
            f"GPT-4 equiv: ${self.total_gpt4_equivalent:.4f} | "
            f"Saved: ${self.total_saved:.4f} ({self.savings_pct:.1f}%)"
        )
