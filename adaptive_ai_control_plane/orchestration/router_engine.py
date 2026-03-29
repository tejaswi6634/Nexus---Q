"""Centralized routing with Q-value tie detection and quantum uncertainty."""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from adaptive_ai_control_plane.ingestion.prompt_request import PromptRequest
from adaptive_ai_control_plane.ingestion.provider_registry import ProviderRegistry
from adaptive_ai_control_plane.orchestration.rl_agent import RLAgent
from adaptive_ai_control_plane.preprocessing.sla_policy_engine import PolicyVerdict

log = logging.getLogger(__name__)

TIE_RELATIVE_THRESHOLD = 0.05  # 5% — top two Q-scores within this band → quantum tie-break


@dataclass
class RoutingEngineOutcome:
    """Result of ``get_routing_decision`` for one step."""

    chosen: str
    reasoning: str
    quantum_randomization_fired: bool
    q_values: Dict[str, float]
    action_index: int


def get_quantum_randomness() -> float:
    """Return a uniform random value in ``[0, 1)``.

    Defaults to a cryptographically strong source (``secrets``). If
    ``IBM_QUANTUM_API_KEY`` or ``QISKIT_IBM_TOKEN`` is set, attempts a
    one-shot IBM Quantum Runtime sampler on real hardware (or Aer if
    ``IBM_QUANTUM_USE_SIMULATOR`` is truthy). Any failure falls back to
    ``secrets``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    log.info("[Quantum Seed Generated] Resolving uncertainty at %s", ts)

    token = os.environ.get("IBM_QUANTUM_API_KEY") or os.environ.get("QISKIT_IBM_TOKEN")
    use_sim = os.environ.get("IBM_QUANTUM_USE_SIMULATOR", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if token:
        try:
            return _ibm_quantum_random_unit_interval(token, use_simulator=use_sim)
        except Exception as exc:  # noqa: BLE001 — demo hook must not break routing
            log.warning("IBM Quantum path failed (%s); using OS CSPRNG", exc)

    return secrets.randbelow(1 << 53) / float(1 << 53)


def _aer_hadamard_bit() -> int:
    """Single-shot H + measure on Aer (local quantum simulator)."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    backend = AerSimulator()
    job = backend.run(transpile(qc, backend), shots=1)
    counts = job.result().get_counts()
    bit_str = max(counts, key=counts.get) if counts else "0"
    return int(bit_str, 2) & 1


def _ibm_quantum_random_unit_interval(token: str, *, use_simulator: bool = False) -> float:
    """IBM hook: Aer when ``use_simulator``; else IBM Quantum Runtime (hardware)."""
    bit: int
    if use_simulator:
        bit = _aer_hadamard_bit()
    else:
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        except ImportError as exc:
            raise RuntimeError("qiskit-ibm-runtime not installed") from exc

        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            backend = service.least_busy(operational=True, simulator=False)
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            sampler = Sampler(mode=backend)
            pub_result = sampler.run([(transpile(qc, backend),)]).result()
            cr = pub_result[0].data.c
            counts = cr.get_counts() if hasattr(cr, "get_counts") else {"0": 1}
            bit_str = max(counts, key=counts.get) if counts else "0"
            bit = int(bit_str, 2) & 1
        except Exception as exc:  # noqa: BLE001
            log.warning("IBM hardware sampler failed (%s); using Aer H⊗|0⟩", exc)
            bit = _aer_hadamard_bit()

    # Map {0,1} + CSPRNG tail to [0, 1)
    tail = secrets.randbelow(1 << 52) / float(1 << 52)
    return (bit * 0.5 + tail * 0.5) % 1.0


def _two_top_within_threshold(scores: Dict[str, float]) -> bool:
    """True only when the two best *learned* scores are relatively tied.

    If the Q-table row is still all zeros (or ~0), we do **not** treat that as a
    meaningful tie — otherwise every early step would trigger quantum tie-break.
    """
    if len(scores) < 2:
        return False
    vals = sorted(scores.values(), reverse=True)
    top, second = vals[0], vals[1]
    mag = max(abs(v) for v in scores.values())
    # No signal yet: uninitialized / flat Q — not a "tie" for quantum purposes
    if mag < 1e-8:
        return False
    denom = abs(top) if abs(top) > 1e-12 else 1.0
    return abs(top - second) / denom <= TIE_RELATIVE_THRESHOLD


def _pick_among_tied(
    ranked_names: List[str],
    scores: Dict[str, float],
) -> str:
    """Break a tie among ``ranked_names`` (best first) using quantum randomness."""
    if len(ranked_names) == 1:
        return ranked_names[0]
    u = get_quantum_randomness()
    # If IBM path returned 0/1 only, spread across list
    if len(ranked_names) > 2:
        idx = int(u * len(ranked_names)) % len(ranked_names)
    else:
        idx = int(u * 2) % 2
    return ranked_names[idx]


def get_routing_decision(
    *,
    state: Tuple[int, int, int, int],
    rl_agent: RLAgent,
    policy: PolicyVerdict,
    pace_status: str,
    request: PromptRequest,
    registry: ProviderRegistry,
    providers: List[str],
) -> RoutingEngineOutcome:
    """Compute the next hop: SLA / burst rules, RL action, optional quantum tie-break.

    When the top two available models have Q-values within 5%, invokes
    :func:`get_quantum_randomness` to choose among the tied leaders.
    """
    if policy.forced_provider:
        q_all = rl_agent.get_scores(state)
        action_idx = next(
            (i for i, p in enumerate(rl_agent.providers) if p == policy.forced_provider),
            0,
        )
        return RoutingEngineOutcome(
            chosen=policy.forced_provider,
            reasoning=f"SLA Policy override → {policy.reason}",
            quantum_randomization_fired=False,
            q_values=q_all,
            action_index=action_idx,
        )

    allowed = [p for p in providers if p not in policy.blocked_providers]
    if not allowed:
        allowed = list(providers)

    if pace_status == "Burst_Mode" and not request.is_high_priority:
        chosen = registry.cheapest_local()
        return RoutingEngineOutcome(
            chosen=chosen,
            reasoning=f"Burst_Mode — shedding low-priority to {chosen}",
            quantum_randomization_fired=False,
            q_values=rl_agent.get_scores(state),
            action_index=next(
                (i for i, p in enumerate(rl_agent.providers) if p == chosen),
                0,
            ),
        )

    action = rl_agent.select_action(state, allowed_providers=allowed)
    chosen = rl_agent.provider_for_action(action)
    scores_full = rl_agent.get_scores(state)
    filtered = {p: scores_full[p] for p in allowed if p in scores_full}

    quantum_fired = False
    reasoning = f"RL Q-Learning (epsilon={rl_agent.epsilon:.3f})"

    if _two_top_within_threshold(filtered):
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]
        denom = abs(top_score) if abs(top_score) > 1e-12 else 1.0
        tied = [
            name
            for name, sc in ranked
            if abs(top_score - sc) / denom <= TIE_RELATIVE_THRESHOLD
        ]
        if len(tied) >= 2:
            chosen = _pick_among_tied(tied, filtered)
            quantum_fired = True
            action = next(
                (i for i, p in enumerate(rl_agent.providers) if p == chosen),
                action,
            )
            rl_agent.override_last_action(action)
            reasoning = (
                f"Quantum tie-break — top Q-values within 5% "
                f"(epsilon={rl_agent.epsilon:.3f})"
            )

    return RoutingEngineOutcome(
        chosen=chosen,
        reasoning=reasoning,
        quantum_randomization_fired=quantum_fired,
        q_values=scores_full,
        action_index=action,
    )
