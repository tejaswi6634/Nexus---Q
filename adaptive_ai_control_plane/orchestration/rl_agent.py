"""Enhanced RL Agent — Q-Learning over (Complexity, Intent, Twin_Health, Pacemaker_Load)."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import random


class RLAgent:
    """Tabular Q-Learning agent whose state is a 4-tuple:

    * **Complexity** — Simple(0), Medium(1), Complex(2)
    * **Intent** — Technical(0), Legal(1), Creative(2), General(3)
    * **Twin_Health** — Healthy(0), Predicted_Degradation(1)
    * **Pacemaker_Load** — Normal(0), Burst_Mode(1)

    Actions map 1-to-1 to the provider list supplied at construction time.
    """

    STATE_DIMS = (3, 4, 2, 2)  # C × I × H × L = 48 states

    def __init__(
        self,
        providers: List[str],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ) -> None:
        self.providers = providers
        self.n_actions = len(providers)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Tiny optimistic spread so actions differ before learning (ties are real, not all-zero)
        rng = np.random.default_rng(42)
        self.q_table: np.ndarray = rng.uniform(
            0.0, 1e-4, size=(*self.STATE_DIMS, self.n_actions)
        ).astype(np.float64)

        self._last_state: Tuple[int, ...] | None = None
        self._last_action: int | None = None
        self.steps: int = 0

    def _validate_state(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        validated = []
        for val, dim in zip(state, self.STATE_DIMS):
            validated.append(max(0, min(val, dim - 1)))
        return tuple(validated)

    def select_action(
        self,
        state: Tuple[int, int, int, int],
        allowed_providers: List[str] | None = None,
    ) -> int:
        state = self._validate_state(state)
        self._last_state = state

        q_values = self.q_table[state].copy()

        if allowed_providers is not None:
            mask = np.full(self.n_actions, -np.inf)
            for i, p in enumerate(self.providers):
                if p in allowed_providers:
                    mask[i] = 0.0
            q_values += mask

        if random.random() < self.epsilon:
            valid = [
                i for i in range(self.n_actions)
                if allowed_providers is None or self.providers[i] in allowed_providers
            ]
            action = random.choice(valid) if valid else int(np.argmax(q_values))
        else:
            action = int(np.argmax(q_values))

        self._last_action = action
        return action

    def override_last_action(self, action_idx: int) -> None:
        """After external tie-break, align the stored action for ``learn()``."""
        if 0 <= action_idx < self.n_actions:
            self._last_action = action_idx

    def get_scores(self, state: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Return Q-values for every provider at the given state."""
        state = self._validate_state(state)
        return {
            p: float(self.q_table[state][i])
            for i, p in enumerate(self.providers)
        }

    def learn(self, reward: float, next_state: Tuple[int, int, int, int]) -> None:
        if self._last_state is None or self._last_action is None:
            return

        next_state = self._validate_state(next_state)
        s, a = self._last_state, self._last_action
        best_next = float(np.max(self.q_table[next_state]))

        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[s][a]
        self.q_table[s][a] += self.alpha * td_error

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

    def provider_for_action(self, action: int) -> str:
        return self.providers[action]
