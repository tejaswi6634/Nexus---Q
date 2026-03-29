"""Quantum Conflict Optimizer — 2-qubit tie-breaking via Qiskit."""

from __future__ import annotations

from typing import Dict, List, Tuple
import logging
import random
import math

logger = logging.getLogger(__name__)

_QISKIT_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    _QISKIT_AVAILABLE = True
except ImportError:
    logger.warning("Qiskit not installed — falling back to classical random tie-breaking.")


class QuantumConflictOptimizer:
    """When the RL agent scores two or more providers within a tight margin,
    this optimizer builds a parametric 2-qubit circuit and uses the measurement
    outcome to break the tie with quantum randomness.

    The circuit encodes normalised score differences as rotation angles so that
    higher-scored providers are *slightly* more likely to be selected while
    still preserving stochastic exploration.
    """

    TIE_THRESHOLD = 0.05  # 5 % relative difference

    def __init__(self, shots: int = 128) -> None:
        self.shots = shots
        self._calls = 0

    def needs_resolution(self, scores: Dict[str, float]) -> bool:
        if len(scores) < 2:
            return False
        vals = sorted(scores.values(), reverse=True)
        top, second = vals[0], vals[1]
        denom = abs(top) if abs(top) > 1e-9 else 1.0
        return abs(top - second) / denom <= self.TIE_THRESHOLD

    def resolve(
        self,
        scores: Dict[str, float],
        allowed: List[str] | None = None,
    ) -> Tuple[str, str]:
        """Select a winner among tied providers.

        Returns (chosen_provider, reasoning_string).
        """
        self._calls += 1

        candidates = {
            k: v for k, v in scores.items()
            if allowed is None or k in allowed
        }
        if not candidates:
            candidates = scores

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]
        denom = abs(top_score) if abs(top_score) > 1e-9 else 1.0
        tied = [
            name for name, sc in ranked
            if abs(top_score - sc) / denom <= self.TIE_THRESHOLD
        ]
        if len(tied) < 2:
            tied = [r[0] for r in ranked[:2]]

        if _QISKIT_AVAILABLE:
            winner = self._quantum_pick(tied, candidates)
            reason = f"Quantum tie-break ({len(tied)} providers within 5%)"
        else:
            winner = self._classical_fallback(tied)
            reason = f"Classical random tie-break (qiskit unavailable, {len(tied)} tied)"

        return winner, reason

    def _quantum_pick(self, tied: List[str], scores: Dict[str, float]) -> str:
        qc = QuantumCircuit(2)

        s0 = scores.get(tied[0], 0.5)
        s1 = scores.get(tied[1 % len(tied)], 0.5)
        total = abs(s0) + abs(s1) if (abs(s0) + abs(s1)) > 1e-9 else 1.0
        theta0 = (s0 / total) * math.pi
        theta1 = (s1 / total) * math.pi

        qc.ry(theta0, 0)
        qc.ry(theta1, 1)
        qc.cx(0, 1)
        qc.h(0)

        sv = Statevector.from_instruction(qc)
        counts = sv.sample_counts(shots=self.shots)

        best_outcome = max(counts, key=counts.get)
        idx = int(best_outcome, 2) % len(tied)
        return tied[idx]

    def _classical_fallback(self, tied: List[str]) -> str:
        return random.choice(tied)

    @property
    def total_calls(self) -> int:
        return self._calls
