"""
HYBRID QUANTUM-CLASSICAL AI ROUTER -- Resilient Inference Pipeline
====================================================================

Production pipeline integrating all core modules **plus** the Advanced
Resilient Inference Engine:

  1. ML-Driven Prompt Complexity Analyzer   (TF-IDF + LogReg)
  2. Provider Benchmark & Telemetry Layer   (ProviderRegistry)
  3. Dynamic Endpoint Health Scorer         (HealthMonitor)
  4. RL-Centric Routing Brain               (DecisionEngine / QLearningAgent)
  5. Post-Response State Synchronisation    (immediate RL update)
  6. Embedding-Based Reward Quality Metric  (Cosine-Similarity reward)

  --- NEW: Resilient Inference Engine Layers ---
  7. Predictive Health Engine               (pre-emptive degradation)
  8. Circuit Breakers                       (death-spiral prevention)
  9. Confidence-Based Quality Gate          (post-response verification)
 10. SLA-Aware Digital Twin Simulator       (simulation-assisted fallback)
 11. Multi-Level Local Fallback Hierarchy   (Llama -> Phi -> SemanticCache)
 12. Graceful Degradation / Prompt Compression

Request flow:
  Predictive Check -> Semantic Cache -> Primary Provider
  -> Confidence Check -> (Optional) Digital Twin Fallback
"""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import adaptive_ai_control_plane.settings  # noqa: F401  # load .env from project root

from ingestion.stream_simulator import StreamSimulator
from ingestion.threadsafe_queue import ThreadSafeQueue
from preprocessing.noise_filter import TokenEstimator, ComplexityClassifier
from preprocessing.feature_engineering import CostPredictor, MetadataEmbeddingLayer
from preprocessing.dimensionality_reduction import DimensionalityReducer
from quantum.executor import QuantumExecutor
from metrics.performance import PerformanceTracker
from metrics.reward import RewardCalculator
from providers.registry import ProviderRegistry
from providers.health_monitor import HealthMonitor
from orchestration.decision_engine import DecisionEngine
from resilience import (
    ResilientRouter,
    UnifiedInferenceRequest,
    UnifiedInferenceResponse,
    UserTier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logger = logging.getLogger("router")

# ---------------------------------------------------------------------
# Simulated provider response -- extended with Local-Phi3-Mini
# ---------------------------------------------------------------------
_PROVIDER_SIM_PROFILES = {
    "OpenAI-GPT4":      {"latency": 0.75, "cost_mult": 1.00, "quality": 0.93, "rate_limit_prob": 0.06},
    "Anthropic-Claude":  {"latency": 0.42, "cost_mult": 0.75, "quality": 0.89, "rate_limit_prob": 0.03},
    "Google-Gemini":     {"latency": 0.35, "cost_mult": 0.50, "quality": 0.84, "rate_limit_prob": 0.02},
    "Local-Llama3-8B":   {"latency": 0.24, "cost_mult": 0.15, "quality": 0.64, "rate_limit_prob": 0.00},
    "Local-Phi3-Mini":   {"latency": 0.12, "cost_mult": 0.08, "quality": 0.52, "rate_limit_prob": 0.00},
}

_PROVIDER_RESPONSE_TEMPLATES = {
    "OpenAI-GPT4":      "Detailed, high-quality analytical response covering all aspects of the request with thorough reasoning.",
    "Anthropic-Claude":  "Clear, balanced response with well-structured analysis addressing the prompt requirements.",
    "Google-Gemini":     "Efficient response with practical focus covering the core requirements concisely.",
    "Local-Llama3-8B":   "Local Llama-3-8B response with solid general knowledge and reasonable detail covering the key aspects of the request.",
    "Local-Phi3-Mini":   "Local Phi-3-Mini distilled response covering essential points concisely with adequate accuracy.",
}


def mock_provider_call(provider_name: str, base_cost: float) -> dict:
    profile = _PROVIDER_SIM_PROFILES.get(
        provider_name, _PROVIDER_SIM_PROFILES["Local-Llama3-8B"]
    )
    rate_limited = random.random() < profile["rate_limit_prob"]
    jitter = random.uniform(-0.05, 0.08)
    latency = max(0.04, profile["latency"] + jitter)
    cost = max(1e-5, base_cost * profile["cost_mult"] * random.uniform(0.9, 1.15))
    quality = max(0.0, min(1.0, profile["quality"] + random.uniform(-0.08, 0.05)))
    success = not (rate_limited and random.random() < 0.8)
    response_text = _PROVIDER_RESPONSE_TEMPLATES.get(
        provider_name, _PROVIDER_RESPONSE_TEMPLATES["Local-Llama3-8B"]
    )
    return {
        "provider": provider_name,
        "latency": latency,
        "cost": cost,
        "quality": quality,
        "success": success,
        "rate_limited": rate_limited,
        "response_text": response_text,
    }


# ---------------------------------------------------------------------
# Convergence plotter
# ---------------------------------------------------------------------
def _plot_convergence(cost_curve, latency_curve, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not found -- skipping convergence plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(cost_curve) + 1)
    ax1.plot(x, cost_curve, label="Cost Savings", color="tab:green")
    ax1.set_xlabel("Prompt Index")
    ax1.set_ylabel("Cost Savings vs Baseline (USD)", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")

    ax2 = ax1.twinx()
    ax2.plot(x, latency_curve, label="Latency", color="tab:blue", alpha=0.8)
    ax2.set_ylabel("Latency (s)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("RL Convergence: Cost Savings vs Latency  (Resilient Engine)")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"[PLOT] Saved convergence chart -> {output_path}")


# ---------------------------------------------------------------------
# Ingestion thread
# ---------------------------------------------------------------------
data_queue: ThreadSafeQueue = ThreadSafeQueue(maxsize=200)


def ingestion_worker(sim: StreamSimulator, q: ThreadSafeQueue) -> None:
    logger.info("Ingestion worker started")
    for packet in sim.stream():
        if not sim.running:
            break
        try:
            q.put(packet, timeout=1.0)
        except q.full_exception():
            logger.warning("Queue full -- dropping request")
    logger.info("Ingestion worker stopped")


# ---------------------------------------------------------------------
# Main pipeline -- Resilient Inference Engine integration
# ---------------------------------------------------------------------
def main() -> None:
    print("=" * 76)
    print("  HYBRID QUANTUM-CLASSICAL AI ROUTER -- Resilient Inference Engine")
    print("  -" * 35)
    print("  Layers: ML Complexity | Provider Telemetry | Health Scorer")
    print("          RL Routing Brain | State Sync | Semantic Reward")
    print("          Predictive Health | Circuit Breakers | Confidence Gate")
    print("          Digital Twin | Local Hierarchy | Prompt Compression")
    print("=" * 76)

    # -- Module 1: ML Complexity Analyzer --------------------------------
    token_estimator = TokenEstimator()
    complexity_clf = ComplexityClassifier()

    # -- Module 2: Provider Registry ------------------------------------
    registry = ProviderRegistry()

    # -- Module 3: Health Monitor ----------------------------------------
    health_monitor = HealthMonitor(registry)

    # -- Module 4: RL Decision Engine (primary router) ------------------
    decision_engine = DecisionEngine(registry, health_monitor)

    # -- Module 6: Embedding-Based Reward Calculator --------------------
    reward_calc = RewardCalculator()

    # -- Supporting layers -----------------------------------------------
    cost_predictor = CostPredictor()
    metadata_embedder = MetadataEmbeddingLayer()
    dim_reducer = DimensionalityReducer(target_dim=4)
    q_executor = QuantumExecutor()
    tracker = PerformanceTracker()

    # -- NEW: Resilient Inference Engine ----------------------------------
    resilient_router = ResilientRouter(
        registry=registry,
        health_monitor=health_monitor,
        decision_engine=decision_engine,
        provider_call_fn=mock_provider_call,
        health_window_seconds=60.0,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
        confidence_threshold=0.40,
        cache_similarity_threshold=0.82,
        high_latency_threshold=1.5,
    )
    resilient_router.start_health_monitoring(interval=5.0)

    # -- Start ingestion thread -------------------------------------------
    initial_rate = 25.0
    stream_sim = StreamSimulator(data_rate_hz=initial_rate)
    stream_sim.running = True
    ingest_thread = threading.Thread(
        target=ingestion_worker, args=(stream_sim, data_queue), daemon=True,
    )
    ingest_thread.start()

    print("\n[SYSTEM] All modules loaded.  Resilient pipeline active.\n")
    hdr = (
        f"{'#':<5} | {'TIME':<9} | {'CX':<7} | "
        f"{'PROVIDER':<20} | {'LAT(s)':<8} | {'COST':<9} | "
        f"{'REWARD':<8} | {'CONF':<6} | {'PRED':<5} | TRACE"
    )
    print(hdr)
    print("-" * len(hdr))

    try:
        total_prompts = 500
        processed = 0
        running_latency = 0.0
        running_cost = 0.0
        baseline_running_cost = 0.0
        latency_curve: list[float] = []
        cost_savings_curve: list[float] = []
        cache_hits = 0
        predicted_failures = 0
        quality_fallbacks = 0

        while processed < total_prompts:
            try:
                packet = data_queue.get(timeout=0.5)
            except data_queue.empty_exception():
                continue

            # -- Phase 1: Feature extraction ----------------------------
            token_count = token_estimator.estimate(
                packet.prompt_text, packet.token_count,
            )
            complexity_label = complexity_clf.classify(packet.prompt_text)
            complexity_score = complexity_clf.score(packet.prompt_text, token_count)

            estimated_cost = cost_predictor.estimate(
                token_count, model_name="Google-Gemini",
            )
            metadata_vector = metadata_embedder.build_vector(
                token_count=token_count,
                complexity_score=complexity_score,
                metadata=packet.metadata,
                estimated_cost=estimated_cost,
            )
            reduced_data = dim_reducer.reduce(metadata_vector)

            # -- Phase 2: Quantum suggestion (advisory) -----------------
            q_result = q_executor.execute_task(
                reduced_data, task_type="model_selection_optimizer",
            )

            # -- Phase 3: Update queue load -----------------------------
            queue_frac = data_queue.qsize() / 200.0
            for pname in registry.provider_names:
                health_monitor.update_queue_load(pname, queue_frac)

            # -- Phase 4: RESILIENT INFERENCE ---------------------------
            user_tier = (
                UserTier.PREMIUM
                if packet.metadata.get("user_priority", 0) >= 0.8
                else UserTier.STANDARD
            )

            inference_request = UnifiedInferenceRequest(
                prompt_text=packet.prompt_text,
                user_tier=user_tier,
                token_count=token_count,
                metadata=packet.metadata,
                complexity_label=complexity_label,
                estimated_cost=estimated_cost,
            )

            response: UnifiedInferenceResponse = resilient_router.infer(
                inference_request,
            )

            selected_model = response.provider
            latency = response.latency
            cost = response.cost
            quality = response.quality
            success = response.success

            # -- Phase 5: Embedding-based reward (Module 6) -------------
            reward = reward_calc.compute(
                actual_response=response.response_text,
                expected_intent=packet.prompt_text,
                latency=latency,
                cost=cost,
            )

            # -- Phase 6: RL state sync ---------------------------------
            is_real_provider = not response.was_cache_hit and "SemanticCache" not in selected_model
            sync_provider = selected_model if is_real_provider else "Local-Llama3-8B"
            resilient_router.synchronise_state(
                provider=sync_provider,
                latency=latency,
                quality=quality,
                cost=cost,
                success=success,
                rate_limited=False,
                reward=reward,
                next_complexity_label=complexity_label,
                next_estimated_cost=estimated_cost,
            )

            # -- Bookkeeping --------------------------------------------
            is_local = selected_model.startswith("Local-") or "SemanticCache" in selected_model
            if is_local:
                tracker.increment_classical()
            else:
                tracker.increment_quantum()

            tracker.log_latency(latency)
            tracker.log_response(
                selected_model=selected_model,
                latency=latency,
                cost=cost,
                success=success,
                reward=reward,
            )

            new_rate, new_dim = decision_engine.adjust_parameters(
                stream_sim.data_rate_hz, dim_reducer.target_dim,
            )
            if new_rate != stream_sim.data_rate_hz:
                stream_sim.data_rate_hz = new_rate
            if new_dim != dim_reducer.target_dim:
                dim_reducer.update_target_dim(new_dim)
            tracker.update_state(new_rate, new_dim)

            processed += 1
            running_latency += latency
            running_cost += cost
            baseline_running_cost += estimated_cost
            latency_curve.append(running_latency / processed)
            cost_savings_curve.append(max(0.0, baseline_running_cost - running_cost))

            if response.was_cache_hit:
                cache_hits += 1
            if response.was_predicted_failure:
                predicted_failures += 1
            if response.verification_provider:
                quality_fallbacks += 1

            if processed % 25 == 0 or processed == 1:
                trace_short = " -> ".join(response.fallback_trace[:3])
                if len(response.fallback_trace) > 3:
                    trace_short += " ->..."
                pred_flag = "PRD" if response.was_predicted_failure else "  -"
                print(
                    f"{processed:<5} | {time.strftime('%H:%M:%S'):<9} | "
                    f"{complexity_label:<7} | {selected_model:<20} | "
                    f"{latency:.4f}  | ${cost:.5f} | "
                    f"{reward:+.4f} | {response.confidence_score:.3f} | "
                    f"{pred_flag:<5} | {trace_short}"
                )

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by user.")
    finally:
        resilient_router.stop_health_monitoring()
        stream_sim.stop()
        ingest_thread.join(timeout=2)

        convergence_path = os.path.join(
            os.path.dirname(__file__), "rl_convergence.png",
        )
        _plot_convergence(cost_savings_curve, latency_curve, convergence_path)

        diag = resilient_router.get_diagnostics()

        summary = tracker.get_summary()
        print("\n" + "=" * 76)
        print("  RESILIENT INFERENCE ENGINE -- FINAL REPORT")
        print("=" * 76)
        print(f"  Total Prompts Processed:  {summary.request_count}")
        print(f"  Cloud-Routed:             {summary.quantum_count}")
        print(f"  Local / Cache Fallback:   {summary.classical_count}")
        print(f"  Avg Latency:              {summary.total_latency:.4f} s")
        print(f"  Total Cost:               ${summary.total_cost:.4f}")
        print(f"  Success Rate:             {tracker.get_success_rate() * 100:.1f}%")
        print(f"  RL Epsilon (final):       {decision_engine.agent.epsilon:.4f}")
        print(f"  Q-Table States:           {len(decision_engine.agent.q_table)}")
        print("  " + "-" * 50)
        print(f"  Semantic Cache Hits:      {cache_hits}")
        print(f"  Predicted Failures:       {predicted_failures}")
        print(f"  Quality Fallbacks:        {quality_fallbacks}")
        print(f"  Cache Size (final):       {diag['cache_size']} entries")
        print(f"  Global Latency EMA:       {diag['global_latency_avg']:.5f} s")
        print("  " + "-" * 50)
        print("  Provider Health (final):")
        for prov, hm in diag["health_metrics"].items():
            print(
                f"    {prov:<20}  status={hm['status']:<8}  "
                f"lat_avg={hm['latency_avg']:.4f}  "
                f"err_rate={hm['error_rate']:.4f}  "
                f"trend={hm['latency_trend_pct']:+.2%}"
            )
        if diag["circuit_breakers"]:
            print("  Circuit Breakers:")
            for prov, state in diag["circuit_breakers"].items():
                print(f"    {prov:<20}  state={state}")
        print("=" * 76)
        print()
        print("  \"Our fallback is not just reactive.")
        print("   It is predictive, confidence-aware, and simulation-assisted.\"")
        print()


if __name__ == "__main__":
    main()
