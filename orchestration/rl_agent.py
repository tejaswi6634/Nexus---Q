"""
Reinforcement Learning Agent for provider routing.

Enhanced Q-Learning agent with:
  - Richer discretised state space (complexity label, health bucket,
    rate-limit flag, cost bucket)
  - Epsilon-decay schedule for exploration → exploitation
  - Immediate post-response state synchronisation (the "rate-limit fix")
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Tabular Q-learning agent whose *actions* correspond to LLM providers.

    Action mapping is injected at construction so the agent stays agnostic
    to the actual provider list (easy to extend).
    """

    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.15,
        discount_factor: float = 0.92,
        epsilon_start: float = 0.25,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.997,
    ) -> None:
        self.n_actions = n_actions
        self.q_table: Dict[Tuple[str, ...], List[float]] = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._steps: int = 0

    def get_state_key(
        self,
        complexity_label: str,
        health_bucket: str,
        rate_limited: bool,
        cost_bucket: str,
    ) -> Tuple[str, ...]:
        """Build a discretised state tuple consumed by the Q-table."""
        rl_flag = "RL_YES" if rate_limited else "RL_NO"
        return (complexity_label, health_bucket, rl_flag, cost_bucket)

    def choose_action(self, state: Tuple[str, ...]) -> int:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def learn(
        self,
        state: Tuple[str, ...],
        action: int,
        reward: float,
        next_state: Tuple[str, ...],
    ) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.n_actions

        predict = self.q_table[state][action]
        target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] += self.lr * (target - predict)

        self._steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
