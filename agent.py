"""
agent.py
========

A tabular **Q-learning** pricing agent -- one per firm.

Each agent is completely independent: it cannot see its rivals' Q-tables,
cannot message them, and only ever observes *prices* and its *own profit*.
Any coordination that emerges therefore emerges purely from repeated
interaction, which is the whole point of PriceWar.

State
-----
An agent's state is a small, discrete summary of "where things stand":

    (own previous price bucket, competitor average price bucket)

Both are indices into the shared price menu, so with a 5-price menu the
state space is 5 x 5 = 25 states.

Action
------
Pick one of the prices from the menu -- i.e. an index in ``range(n_prices)``.

Reward
------
The firm's own profit for the round.

Update
------
The textbook Q-learning rule::

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

from __future__ import annotations

import numpy as np

from utils import nearest_index


class QLearningAgent:
    """One independent tabular Q-learning firm."""

    def __init__(
        self,
        firm_id: int,
        price_set,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.firm_id = firm_id
        self.price_set = np.asarray(price_set, dtype=float)
        self.n_prices = len(self.price_set)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = rng if rng is not None else np.random.default_rng()

        # Q-table indexed by [own_price_idx, competitor_price_idx, action].
        # Initialised to zero -- optimistic initialisation is unnecessary here
        # because the epsilon schedule already drives plenty of exploration.
        self.q = np.zeros((self.n_prices, self.n_prices, self.n_prices))

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------
    def encode_state(self, own_price: float, competitor_avg_price: float):
        """Map raw prices to a ``(own_idx, comp_idx)`` state tuple."""
        own_idx = nearest_index(own_price, self.price_set)
        comp_idx = nearest_index(competitor_avg_price, self.price_set)
        return own_idx, comp_idx

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def select_action(self, state) -> int:
        """Epsilon-greedy action selection.

        With probability ``epsilon`` explore a random price; otherwise exploit
        the best price the Q-table currently knows about.  Ties are broken
        randomly so the agent isn't biased toward low indices early on.
        """
        own_idx, comp_idx = state
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_prices))
        q_values = self.q[own_idx, comp_idx]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def price_for_action(self, action: int) -> float:
        return float(self.price_set[action])

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, state, action: int, reward: float, next_state) -> None:
        """Apply one Q-learning update."""
        own_idx, comp_idx = state
        n_own, n_comp = next_state
        best_next = np.max(self.q[n_own, n_comp])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[own_idx, comp_idx, action]
        self.q[own_idx, comp_idx, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Shrink exploration toward the floor after every step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Introspection helpers (used by the dashboard / explanation layer)
    # ------------------------------------------------------------------
    def greedy_policy(self) -> np.ndarray:
        """Best-known price index for every state -- shape ``(n_prices, n_prices)``."""
        return np.argmax(self.q, axis=2)

    def value_surface(self) -> np.ndarray:
        """Max Q-value per state, for a heatmap -- shape ``(n_prices, n_prices)``."""
        return np.max(self.q, axis=2)
