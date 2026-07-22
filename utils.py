"""
utils.py
========

Helper functions and the central configuration object for PriceWar.

Keeping the configuration in one place (``SimulationConfig``) means the
market, the agents, the simulation loop and the Streamlit dashboard all speak
the same language.  Nothing here knows about reinforcement learning or
economics specifically -- these are small, reusable building blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """All tunable knobs for a PriceWar experiment.

    The defaults below produce interesting, non-trivial dynamics: firms start
    out undercutting each other and then (often) drift upward toward higher,
    more profitable prices -- the tacit-coordination story that PriceWar is
    designed to illustrate.
    """

    # --- Market structure -------------------------------------------------
    n_firms: int = 2
    #: The discrete menu of prices every firm can choose from.
    prices: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0])
    #: Marginal cost per firm.  If a single float is given it is broadcast to
    #: every firm; a list lets you give firms *different* costs.
    costs: List[float] = field(default_factory=lambda: [4.0, 4.0])

    # --- Demand model -----------------------------------------------------
    #: Market size when the average price is zero (the intercept ``a``).
    base_demand: float = 200.0
    #: How quickly total demand shrinks as the average price rises (slope ``b``).
    sensitivity: float = 5.0
    #: Competition intensity for the market-share softmax.  Higher = customers
    #: react more strongly to price *differences* between firms.
    beta: float = 0.10

    # --- Q-learning hyper-parameters -------------------------------------
    alpha: float = 0.15          #: learning rate
    gamma: float = 0.95          #: discount factor
    epsilon: float = 1.0         #: initial exploration probability
    epsilon_min: float = 0.01    #: floor for epsilon
    #: Multiplicative decay applied to epsilon each step.  Leave as ``None``
    #: to auto-tune it to the run length (epsilon reaches its floor around 70%
    #: of the way through), or set an explicit value to control it by hand.
    epsilon_decay: float | None = None

    # --- Simulation -------------------------------------------------------
    n_steps: int = 1500
    seed: int = 42

    # ------------------------------------------------------------------
    def effective_epsilon_decay(self) -> float:
        """The per-step epsilon decay actually used by the simulation.

        When ``epsilon_decay`` is left as ``None`` we solve for the decay that
        drives epsilon from its start value down to ``epsilon_min`` at roughly
        70% of the run -- so that whatever run length the user picks, the final
        stretch is (near-)greedy and prices get a chance to settle.
        """
        if self.epsilon_decay is not None:
            return float(self.epsilon_decay)
        target_step = max(1, int(0.7 * self.n_steps))
        ratio = max(self.epsilon_min, 1e-6) / max(self.epsilon, 1e-6)
        return float(ratio ** (1.0 / target_step))

    # ------------------------------------------------------------------
    def normalized_costs(self) -> List[float]:
        """Return a cost value for *every* firm.

        Accepts either a single number (broadcast to all firms) or a list.
        A list shorter than ``n_firms`` is padded with its last value; a
        longer list is truncated.
        """
        if isinstance(self.costs, (int, float)):
            return [float(self.costs)] * self.n_firms
        costs = list(self.costs)
        if len(costs) < self.n_firms:
            costs = costs + [costs[-1]] * (self.n_firms - len(costs))
        return [float(c) for c in costs[: self.n_firms]]

    def n_prices(self) -> int:
        return len(self.prices)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def nearest_index(value: float, grid: Sequence[float]) -> int:
    """Index of the entry in ``grid`` closest to ``value``.

    Used to *discretize* a continuous competitor price into one of the buckets
    that make up an agent's state.
    """
    grid_arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(grid_arr - value)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def moving_average(series: Sequence[float], window: int) -> np.ndarray:
    """Simple trailing moving average, same length as the input.

    The first ``window - 1`` points are averaged over however many samples
    exist so far, so the output has no NaNs and lines up with the raw series
    on a chart.
    """
    arr = np.asarray(series, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    out = np.empty_like(arr)
    cumsum = np.cumsum(arr)
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        if lo == 0:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[lo - 1]) / (i - lo + 1)
    return out


def price_stability(series: Sequence[float], window: int = 100) -> float:
    """Standard deviation of the last ``window`` values of a series.

    A value near zero means prices have stopped moving -- our practical
    definition of "convergence".
    """
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return float("nan")
    tail = arr[-window:]
    return float(np.std(tail))
