import time
import random
from typing import Generator, List, Dict, Any

from .prompt_request import PromptRequest

class StreamSimulator:
    """
    Simulates incoming user prompts for the quantum-classical LLM router.
    """

    def __init__(self, data_rate_hz: float = 10.0):
        self.data_rate_hz = data_rate_hz
        self.running = False
        self._prompt_bank: List[str] = [
            "Summarize this contract and list legal risks.",
            "Write a short marketing email for a new mobile app launch.",
            "Generate Python code for a rate limiter with tests.",
            "Explain quantum angle encoding with a practical example.",
            "Convert meeting notes into action items and owners.",
            "Draft a SQL query to find top churned users by region.",
            "Rewrite this paragraph for a non-technical audience.",
            "Design a low-latency API architecture with tradeoffs.",
            "Translate this message into Spanish and French.",
            "Create a concise incident postmortem from raw logs.",
        ]

    def _estimate_token_count(self, prompt_text: str) -> int:
        # Simple synthetic estimator for simulator-only generation.
        word_count = max(1, len(prompt_text.split()))
        return int(word_count * random.uniform(1.1, 1.8))

    def _build_metadata(self) -> Dict[str, Any]:
        return {
            "user_priority": random.choice([0.2, 0.5, 0.8, 1.0]),
            "model_constraints": {
                "max_budget_usd": random.choice([0.001, 0.003, 0.006, 0.01]),
                "latency_target_ms": random.choice([300, 700, 1200]),
            },
        }

    def stream(self) -> Generator[PromptRequest, None, None]:
        self.running = True
        interval = 1.0 / self.data_rate_hz

        while self.running:
            start_time = time.time()
            prompt_text = random.choice(self._prompt_bank)
            metadata = self._build_metadata()
            token_count = self._estimate_token_count(prompt_text)
            yield PromptRequest(
                prompt_text=prompt_text,
                metadata=metadata,
                token_count=token_count,
                timestamp=time.time(),
            )

            elapsed = time.time() - start_time
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False
