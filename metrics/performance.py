import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SystemMetrics:
    total_latency: float = 0.0
    quantum_count: int = 0
    classical_count: int = 0
    current_sampling_rate: float = 0.0
    current_dim: int = 0
    total_cost: float = 0.0
    success_count: int = 0
    request_count: int = 0

class PerformanceTracker:
    """
    Tracks runtime metrics to feed into the orchestration layer.
    """
    def __init__(self):
        self.metrics = SystemMetrics()
        self.latencies = []
        self.response_log: List[Dict[str, Any]] = []

    def log_latency(self, latency: float):
        self.latencies.append(latency)
        # Keep window small for rolling average
        if len(self.latencies) > 50:
            self.latencies.pop(0)
        self.metrics.total_latency = sum(self.latencies) / len(self.latencies)

    def increment_quantum(self):
        self.metrics.quantum_count += 1

    def increment_classical(self):
        self.metrics.classical_count += 1

    def log_response(
        self,
        selected_model: str,
        latency: float,
        cost: float,
        success: bool,
        reward: float,
    ):
        self.metrics.request_count += 1
        self.metrics.total_cost += max(0.0, cost)
        if success:
            self.metrics.success_count += 1
        self.response_log.append(
            {
                "timestamp": time.time(),
                "selected_model": selected_model,
                "latency": latency,
                "cost": cost,
                "success": success,
                "reward": reward,
            }
        )
        
    def update_state(self, rate, dim):
        self.metrics.current_sampling_rate = rate
        self.metrics.current_dim = dim

    def get_summary(self):
        return self.metrics

    def get_success_rate(self) -> float:
        if self.metrics.request_count == 0:
            return 0.0
        return self.metrics.success_count / self.metrics.request_count
