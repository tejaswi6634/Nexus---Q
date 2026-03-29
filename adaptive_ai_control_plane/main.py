"""Nexus-Q — Execution, State Synchronization & Simulation Loop.

CLI:
    cd hybrid_qc_framework
    python -m adaptive_ai_control_plane.main

Interactive cockpit (FastAPI + WebSocket):
    python -m adaptive_ai_control_plane web
    Open http://127.0.0.1:8765
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from adaptive_ai_control_plane import settings
from adaptive_ai_control_plane.claude_client import complete_claude
from adaptive_ai_control_plane.gemini_client import complete_gemini
from adaptive_ai_control_plane.mistral_client import complete_mistral
from adaptive_ai_control_plane.openai_client import complete_openai
from adaptive_ai_control_plane.ingestion.prompt_request import (
    PromptRequest,
    Priority,
    RoutingDecision,
)
from adaptive_ai_control_plane.ingestion.provider_registry import ProviderRegistry
from adaptive_ai_control_plane.preprocessing.complexity_classifier import ComplexityClassifier
from adaptive_ai_control_plane.preprocessing.intent_detector import IntentDetector
from adaptive_ai_control_plane.preprocessing.priority_classifier import PriorityClassifier
from adaptive_ai_control_plane.preprocessing.sla_policy_engine import SLAPolicyEngine
from adaptive_ai_control_plane.digital_twin.simulation_engine import SimulationEngine
from adaptive_ai_control_plane.digital_twin.predictive_logic import PredictiveLogic
from adaptive_ai_control_plane.digital_twin.cost_twin import CostTwin
from adaptive_ai_control_plane.pacemaker.flow_controller import FlowController
from adaptive_ai_control_plane.orchestration.rl_agent import RLAgent
from adaptive_ai_control_plane.orchestration.router_engine import get_routing_decision
from adaptive_ai_control_plane.metrics.reward_calculator import RewardCalculator
from adaptive_ai_control_plane.finance.savings_tracker import SavingsTracker
from anthropic import APIStatusError, RateLimitError
from openai import APIError, RateLimitError as OpenAIRateLimitError

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("Nexus-Q")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Corpus
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_POOL: List[str] = [
    # 0-9  — mixed traffic
    "What is the capital of France?",
    "Write a Python function for binary search with edge-case handling",
    "Draft a GDPR compliance review for our data-pipeline architecture",
    "Design a microservices architecture for an e-commerce platform with event sourcing and CQRS",
    "Tell me a creative story about space exploration and alien civilizations",
    "Explain quantum computing basics and qubit entanglement",
    "Review this contract clause for legal compliance with international trade law",
    "Optimize this SQL query for better performance across partitioned tables",
    "Write a poem about artificial intelligence and the future of humanity",
    "Implement a distributed consensus algorithm with Byzantine fault tolerance",
    # 10-14
    "What is machine learning?",
    "Create a comprehensive security audit framework for a multi-cloud Kubernetes deployment",
    "How do neural networks learn from data?",
    "Draft an intellectual property licensing agreement for our open-source project",
    "Build a real-time recommendation engine with collaborative filtering",
    # 15-19
    "Write a haiku about programming",
    "Explain the CAP theorem and its implications for distributed databases",
    "Design an end-to-end MLOps pipeline with model versioning and canary deployments",
    "What are the current CCPA regulations for handling consumer personal data?",
    "Implement a custom memory allocator with garbage collection for a language runtime",
    # 20-24
    "Summarize the history of the Internet in three sentences",
    "Architect a zero-trust security framework for hybrid-cloud environments",
    "Write creative marketing copy for a new AI product launch campaign",
    "Explain how the TCP three-way handshake works in network protocols",
    "What is recursion?",
    # 25-39  — burst / spike filler
    "Quick health-check ping",
    "Status report request",
    "Simple lookup query for user data",
    "Check database connection status",
    "Retrieve cached configuration values",
    "Log aggregation summary for today",
    "Quick metric snapshot request",
    "Heartbeat verification ping",
    "System uptime query",
    "Cache invalidation request",
    "Ping the load balancer",
    "Fetch API version information",
    "Quick DNS lookup request",
    "Health endpoint response check",
    "Service discovery query",
    # 40-49  — post-burst recovery
    "Design a privacy-preserving federated learning system with differential privacy",
    "Write a creative screenplay outline for a science-fiction thriller",
    "Analyze legal implications of deploying AI in healthcare diagnostics",
    "Build a quantum-resistant encryption protocol for enterprise communications",
    "Create an interactive data-visualization dashboard for real-time analytics",
    "How does a hash table handle collisions?",
    "Implement a distributed tracing system for microservices observability",
    "Write a children's bedtime story about a robot learning to paint",
    "Design a multi-modal AI inference serving platform with dynamic batching and GPU management",
    "What are the three laws of thermodynamics?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Response Generator
# ─────────────────────────────────────────────────────────────────────────────

_FILLER_TOKENS = [
    "The", "answer", "involves", "considering", "multiple", "factors",
    "including", "performance", "scalability", "reliability", "cost",
    "efficiency", "in", "modern", "systems", "architecture", "leveraging",
    "advanced", "techniques", "for", "optimal", "results", "detailed",
    "analysis", "comprehensive", "approach", "implementation", "strategy",
]


def _simulate_response_text(prompt: str, quality: float) -> str:
    """Generate synthetic response text whose lexical overlap with the prompt
    is proportional to the provider's quality score."""
    words = prompt.split()
    n_keep = max(1, int(len(words) * quality))
    kept = random.sample(words, min(n_keep, len(words)))
    filler = random.sample(_FILLER_TOKENS, min(8, len(_FILLER_TOKENS)))
    parts = kept + filler
    random.shuffle(parts)
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# State Synchronizer
# ─────────────────────────────────────────────────────────────────────────────

class StateSynchronizer:
    """Post-response state synchronization across every subsystem.

    After a provider responds this class ensures:
    1. The **Provider Registry** updates its telemetry window.
    2. The **Digital Twin** records the observation for trend prediction.
    3. The **Cost Twin** logs the spend.
    4. The **Reward Calculator** scores the interaction.
    5. The **RL Agent** executes its ``learn()`` step.
    6. The **Savings Tracker** records the financial delta.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        twin_engine: SimulationEngine,
        rl_agent: RLAgent,
        cost_twin: CostTwin,
        reward_calc: RewardCalculator,
        savings: SavingsTracker,
    ) -> None:
        self.registry = registry
        self.twin_engine = twin_engine
        self.rl = rl_agent
        self.cost_twin = cost_twin
        self.reward_calc = reward_calc
        self.savings = savings

    def sync(
        self,
        provider: str,
        prompt: str,
        response_text: str,
        latency: float,
        cost: float,
        quality: float,
        complexity: str,
        next_state: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """Run the full synchronization cycle.  Returns ``(reward, saved)``."""
        self.registry.update_telemetry(provider, latency)
        self.twin_engine.record(provider, latency, cost, quality, time.time())
        self.cost_twin.record_cost(cost)
        reward = self.reward_calc.compute(prompt, response_text, latency, cost)
        self.rl.learn(reward, next_state)
        gpt4_cost = self.registry.gpt4_equivalent_cost(complexity)
        saved = self.savings.record(cost, gpt4_cost)
        return reward, saved


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration Loop
# ─────────────────────────────────────────────────────────────────────────────


def iter_simulation_events(
    n_requests: int = 50,
    daily_budget: float = 1.00,
    *,
    user_prompts: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield JSON-friendly events for each simulation step (CLI or Web UI).

    If ``user_prompts`` is non-empty, each step cycles through that list
    (repeating as needed for ``n_requests``).
    Otherwise the built-in ``PROMPT_POOL`` drives traffic.

    Priority is always **auto-detected** by the ``PriorityClassifier`` — no
    manual annotation required.
    """
    registry = ProviderRegistry()
    classifier = ComplexityClassifier()
    intent_det = IntentDetector()
    priority_clf = PriorityClassifier()
    sla_engine = SLAPolicyEngine()
    twin_engine = SimulationEngine()
    predictive = PredictiveLogic(twin_engine)
    cost_twin = CostTwin(daily_budget, twin_engine)
    pacemaker = FlowController(burst_threshold=10, window_sec=5.0, cooldown_sec=8.0)
    rl_agent = RLAgent(providers=registry.provider_names())
    reward_calc = RewardCalculator()
    savings = SavingsTracker()

    sync = StateSynchronizer(
        registry, twin_engine, rl_agent, cost_twin, reward_calc, savings,
    )

    providers = registry.provider_names()

    use_custom = bool(user_prompts)
    up = [p for p in (user_prompts or []) if (p or "").strip()]
    if use_custom and not up:
        use_custom = False

    yield {
        "type": "start",
        "n_requests": n_requests,
        "daily_budget": daily_budget,
        "providers": list(providers),
        "prompt_source": "user" if use_custom else "corpus",
        "user_prompt_variants": len(up) if use_custom else 0,
        "priority_mode": "auto-detected",
        "claude_live_configured": bool(settings.ANTHROPIC_API_KEY),
        "openai_live_configured": bool(settings.OPENAI_API_KEY),
        "google_live_configured": bool(settings.GOOGLE_API_KEY),
        "mistral_live_configured": bool(settings.MISTRAL_API_KEY),
    }

    rate_limit_events = 0
    quantum_resolutions = 0

    for i in range(n_requests):
        if use_custom:
            prompt_text = up[i % len(up)]
        else:
            prompt_text = PROMPT_POOL[i % len(PROMPT_POOL)]

        priority, priority_conf = priority_clf.classify_with_confidence(prompt_text)

        request = PromptRequest(
            text=prompt_text, priority=priority, budget_limit=daily_budget,
        )

        is_spike = 25 <= i < 40
        if is_spike:
            for _ in range(3):
                pacemaker.heartbeat()
                time.sleep(0.008)

        pacemaker.heartbeat()
        pace_status = pacemaker.mode

        complexity, conf = classifier.classify_with_confidence(prompt_text)
        intent = intent_det.detect(prompt_text)
        c_idx = classifier.complexity_index(prompt_text)
        i_idx = intent_det.intent_index(prompt_text)

        worst_health = 0
        twin_report: dict[str, str] = {}
        for p in providers:
            status = predictive.health_status(p)
            twin_report[p] = status
            if status == "Predicted_Degradation":
                worst_health = 1

        policy = sla_engine.evaluate(request, intent, cost_twin.remaining)

        pace_load = pacemaker.load_index
        state = (c_idx, i_idx, worst_health, pace_load)

        route_out = get_routing_decision(
            state=state,
            rl_agent=rl_agent,
            policy=policy,
            pace_status=pace_status,
            request=request,
            registry=registry,
            providers=providers,
        )
        chosen = route_out.chosen
        reasoning = route_out.reasoning
        quantum_used = route_out.quantum_randomization_fired
        if quantum_used:
            quantum_resolutions += 1

        inject_deg = is_spike and chosen in ("GPT-4", "Claude-3", "Gemini-1.5")
        profile = registry.get(chosen)

        use_live_openai = (
            chosen == "GPT-4"
            and settings.should_use_live_openai(use_custom)
        )
        use_live_claude = (
            chosen == "Claude-3"
            and settings.should_use_live_claude(use_custom)
        )
        use_live_gemini = (
            chosen == "Gemini-1.5"
            and settings.should_use_live_google(use_custom)
        )
        use_live_mistral = (
            chosen == "Mistral-7B"
            and settings.should_use_live_mistral(use_custom)
        )

        response_text = ""
        latency = 0.0
        cost = 0.0
        quality = 0.0
        inference_mode = "simulated"

        if use_live_openai:
            qh = profile.quality_score
            try:
                response_text, latency, cost, quality = complete_openai(
                    prompt_text,
                    quality_hint=qh,
                )
                inference_mode = "live_openai"
            except OpenAIRateLimitError as e:
                log.warning("OpenAI rate limit: %s", e)
                rate_limit_events += 1
                fallback = registry.cheapest_local()
                reasoning += (
                    f" | HTTP 429 on GPT-4 (live) → fallback to {fallback}"
                )
                chosen = fallback
                profile = registry.get(chosen)
                latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)
            except APIError as e:
                sc = getattr(e, "status_code", None)
                log.warning(
                    "OpenAI API error %s: %s — using simulation",
                    sc,
                    e,
                )
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | HTTP 429 on {chosen} → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)
            except Exception as e:
                log.warning("OpenAI request failed: %s — using simulation", e)
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | HTTP 429 on {chosen} → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)

        elif use_live_claude:
            qh = profile.quality_score
            try:
                response_text, latency, cost, quality = complete_claude(
                    prompt_text,
                    quality_hint=qh,
                )
                inference_mode = "live_anthropic"
            except RateLimitError as e:
                log.warning("Anthropic rate limit: %s", e)
                rate_limit_events += 1
                fallback = registry.cheapest_local()
                reasoning += (
                    f" | HTTP 429 on Claude (live) → fallback to {fallback}"
                )
                chosen = fallback
                profile = registry.get(chosen)
                latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)
            except APIStatusError as e:
                sc = e.status_code
                log.warning(
                    "Anthropic API error %s: %s — using simulation",
                    sc,
                    e,
                )
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | HTTP 429 on {chosen} → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)
            except Exception as e:
                log.warning("Claude request failed: %s — using simulation", e)
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | HTTP 429 on {chosen} → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)

        elif use_live_gemini:
            qh = profile.quality_score
            try:
                response_text, latency, cost, quality = complete_gemini(
                    prompt_text,
                    quality_hint=qh,
                )
                inference_mode = "live_google"
            except Exception as e:
                log.warning("Gemini request failed: %s — using simulation", e)
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | Rate limit on Gemini (live) → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)

        elif use_live_mistral:
            qh = profile.quality_score
            try:
                response_text, latency, cost, quality = complete_mistral(
                    prompt_text,
                    quality_hint=qh,
                )
                inference_mode = "live_mistral"
            except OpenAIRateLimitError as e:
                log.warning("Mistral rate limit: %s", e)
                rate_limit_events += 1
                fallback = registry.cheapest_local()
                reasoning += (
                    f" | HTTP 429 on Mistral (live) → fallback to {fallback}"
                )
                chosen = fallback
                profile = registry.get(chosen)
                latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)
            except APIError as e:
                sc = getattr(e, "status_code", None)
                log.warning(
                    "Mistral API error %s: %s — using simulation",
                    sc,
                    e,
                )
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | HTTP 429 on {chosen} → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)
            except Exception as e:
                log.warning("Mistral request failed: %s — using simulation", e)
                latency, cost, quality, rate_limited = profile.simulate_response(
                    complexity, inject_degradation=inject_deg,
                )
                if rate_limited:
                    rate_limit_events += 1
                    fallback = registry.cheapest_local()
                    reasoning += (
                        f" | HTTP 429 on {chosen} → fallback to {fallback}"
                    )
                    chosen = fallback
                    profile = registry.get(chosen)
                    latency, cost, quality, _ = profile.simulate_response(complexity)
                response_text = _simulate_response_text(prompt_text, quality)

        if not response_text:
            latency, cost, quality, rate_limited = profile.simulate_response(
                complexity, inject_degradation=inject_deg,
            )
            if rate_limited:
                rate_limit_events += 1
                fallback = registry.cheapest_local()
                reasoning += f" | HTTP 429 on {chosen} → fallback to {fallback}"
                chosen = fallback
                profile = registry.get(chosen)
                latency, cost, quality, _ = profile.simulate_response(complexity)
            response_text = _simulate_response_text(prompt_text, quality)
            inference_mode = "simulated"

        preview = response_text
        if len(preview) > 800:
            preview = preview[:800] + "…"
        twin_pred = twin_report.get(chosen, "Healthy")

        next_worst = max(
            (1 if predictive.health_status(p) == "Predicted_Degradation" else 0)
            for p in providers
        )
        next_state = (c_idx, i_idx, next_worst, pace_load)

        reward, saved = sync.sync(
            provider=chosen,
            prompt=prompt_text,
            response_text=response_text,
            latency=latency,
            cost=cost,
            quality=quality,
            complexity=complexity,
            next_state=next_state,
        )

        latency_hist = registry.latency_history_snapshot(providers, n=8)
        explainability = {
            "state_vector": {
                "prompt_complexity": complexity,
                "provider_health": dict(twin_report),
                "latency_history": latency_hist,
            },
            "blocked_providers": list(policy.blocked_providers),
            "q_values": {k: float(v) for k, v in route_out.q_values.items()},
            "quantum_randomization_fired": route_out.quantum_randomization_fired,
        }

        yield {
            "type": "step",
            "index": i + 1,
            "prompt": prompt_text,
            "priority": priority.value,
            "priority_confidence": priority_conf,
            "is_spike": is_spike,
            "complexity": complexity,
            "confidence": conf,
            "intent": intent,
            "pacemaker": pace_status,
            "pace_load": pace_load,
            "twin_report": dict(twin_report),
            "explainability": explainability,
            "provider": chosen,
            "model": {
                "live_openai": settings.OPENAI_MODEL,
                "live_anthropic": settings.ANTHROPIC_MODEL,
                "live_google": settings.GOOGLE_MODEL,
                "live_mistral": settings.MISTRAL_MODEL,
            }.get(inference_mode, chosen),
            "inference_mode": inference_mode,
            "response_text": response_text,
            "response_preview": preview,
            "reasoning": reasoning,
            "twin_prediction": twin_pred,
            "quantum_resolved": quantum_used,
            "latency": latency,
            "cost": cost,
            "quality": quality,
            "reward": reward,
            "saved": saved,
            "budget_remaining": cost_twin.remaining,
            "budget_spent": cost_twin.spent,
            "rl_epsilon": rl_agent.epsilon,
            "rl_steps": rl_agent.steps,
        }

    yield {
        "type": "complete",
        "savings_summary": savings.summary(),
        "rate_limit_events": rate_limit_events,
        "quantum_resolutions": quantum_resolutions,
        "rl_steps": rl_agent.steps,
        "rl_epsilon": rl_agent.epsilon,
        "budget_spent": cost_twin.spent,
        "budget_limit": daily_budget,
        "utilization_pct": cost_twin.utilization_pct,
    }


def run_simulation(n_requests: int = 50, daily_budget: float = 1.00) -> None:
    for event in iter_simulation_events(n_requests, daily_budget):
        et = event["type"]
        if et == "start":
            log.info("=" * 80)
            log.info("  Nexus-Q  —  Autonomous Control-Plane Simulation")
            log.info(
                f"  Requests : {event['n_requests']}  |  "
                f"Daily Budget : ${event['daily_budget']:.2f}"
            )
            log.info(f"  Providers: {', '.join(event['providers'])}")
            src = event.get("prompt_source", "corpus")
            log.info(f"  Prompts : {src}" + (
                f" ({event.get('user_prompt_variants', 0)} variant(s))"
                if src == "user" else " (built-in corpus)"
            ))
            log.info(f"  Priority: auto-detected by ML classifier")
            if event.get("openai_live_configured"):
                log.info(
                    "  OpenAI live API: configured (used only with user-supplied prompts)"
                )
            else:
                log.info("  OpenAI live API: not configured (set OPENAI_API_KEY in .env)")
            if event.get("claude_live_configured"):
                log.info(
                    "  Claude live API: configured (used only with user-supplied prompts)"
                )
            else:
                log.info("  Claude live API: not configured (set ANTHROPIC_API_KEY in .env)")
            if event.get("google_live_configured"):
                log.info(
                    "  Gemini live API: configured (used only with user-supplied prompts)"
                )
            else:
                log.info("  Gemini live API: not configured (set GOOGLE_API_KEY in .env)")
            if event.get("mistral_live_configured"):
                log.info(
                    "  Mistral live API: configured (used only with user-supplied prompts)"
                )
            else:
                log.info(
                    "  Mistral live API: not configured (set MISTRAL_API_KEY in .env)"
                )
            log.info("=" * 80)
        elif et == "step":
            decision = RoutingDecision(
                provider=event["provider"],
                twin_prediction=event["twin_prediction"],
                pacemaker_status=event["pacemaker"],
                reasoning=event["reasoning"],
                confidence=event["confidence"],
                estimated_cost=event["cost"],
                estimated_latency=event["latency"],
                quantum_resolved=event["quantum_resolved"],
            )
            tag = " [TRAFFIC SPIKE]" if event["is_spike"] else ""
            pt = event["prompt"]
            log.info(f"\n{'─' * 80}")
            log.info(
                f"  Request #{event['index']:>3}{tag}  |  "
                f"Priority: {event['priority'].upper()} "
                f"(auto-detected, {event['priority_confidence']:.0%})"
            )
            log.info(
                f"  Prompt   : {pt[:72]}{'…' if len(pt) > 72 else ''}"
            )
            log.info(
                f"  Analysis : Complexity={event['complexity']} "
                f"({event['confidence']:.0%})  "
                f"Intent={event['intent']}  Pacemaker={event['pacemaker']}"
            )
            log.info(f"  {decision.explain(roi=event['saved'])}")
            log.info(
                f"  Metrics  : Latency={event['latency']:.3f}s  "
                f"Cost=${event['cost']:.6f}  "
                f"Reward={event['reward']:.4f}  "
                f"Budget_Left=${event['budget_remaining']:.4f}"
            )
            im = event.get("inference_mode")
            if im in ("live_openai", "live_anthropic", "live_google", "live_mistral"):
                prev = event.get("response_text") or event.get("response_preview") or ""
                if prev:
                    label = {
                        "live_openai": "GPT (live)",
                        "live_anthropic": "Claude (live)",
                        "live_google": "Gemini (live)",
                        "live_mistral": "Mistral (live)",
                    }[im]
                    log.info(
                        f"  {label}: {prev[:220]}{'…' if len(prev) > 220 else ''}"
                    )
        else:
            log.info(f"\n{'═' * 80}")
            log.info("  SIMULATION COMPLETE")
            log.info(f"{'═' * 80}")
            log.info(f"  {event['savings_summary']}")
            log.info(f"  Rate-limit (429) events : {event['rate_limit_events']}")
            log.info(
                f"  Quantum tie-breaks      : {event['quantum_resolutions']}"
            )
            log.info(
                f"  RL episodes             : {event['rl_steps']}  |  "
                f"Final epsilon: {event['rl_epsilon']:.4f}"
            )
            log.info(
                f"  Budget utilisation      : ${event['budget_spent']:.4f} / "
                f"${event['budget_limit']:.2f}  "
                f"({event['utilization_pct']:.1f}%)"
            )
            log.info(f"{'═' * 80}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_simulation(n_requests=50, daily_budget=1.00)
