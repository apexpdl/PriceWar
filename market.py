"""
market.py
=========

The economic environment for PriceWar.

The :class:`Market` turns a vector of prices (one per firm) into demand,
market share and profit.  It is deliberately small and transparent so the
economics stay legible:

* **Total demand** falls linearly as the *average* price rises
  (the classic downward-sloping demand curve).
* **Market share** is allocated with a softmax on price, so a firm that
  undercuts its rivals wins a larger slice of the pie (Bertrand-style
  business stealing) without any single firm ever capturing 100%.
* **Profit** for a firm is ``(price - cost) * quantity_sold``.

The class also knows how to compute two textbook benchmarks on the discrete
price grid -- the competitive **Nash** price and the joint-profit-maximising
**collusive** price -- which the explanation layer uses to decide whether the
agents ended up competing or (tacitly) coordinating.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np

from utils import softmax


@dataclass
class MarketOutcome:
    """Result of a single market interaction."""

    prices: np.ndarray        #: price chosen by each firm
    shares: np.ndarray        #: fraction of total demand each firm captured
    demands: np.ndarray       #: quantity sold by each firm
    profits: np.ndarray       #: profit earned by each firm
    total_demand: float       #: size of the whole market this round
    avg_price: float          #: average price across firms


class Market:
    """A differentiated-goods price-competition environment.

    Parameters mirror :class:`utils.SimulationConfig` but are passed
    explicitly so :class:`Market` can be used and tested on its own.
    """

    def __init__(
        self,
        prices: Sequence[float],
        costs: Sequence[float],
        base_demand: float = 200.0,
        sensitivity: float = 5.0,
        beta: float = 0.10,
    ) -> None:
        self.price_set = np.asarray(prices, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.base_demand = float(base_demand)
        self.sensitivity = float(sensitivity)
        self.beta = float(beta)
        self.n_firms = len(self.costs)

    # ------------------------------------------------------------------
    # Core economic primitives
    # ------------------------------------------------------------------
    def total_demand(self, avg_price: float) -> float:
        """Downward-sloping linear demand: ``Q = max(0, a - b * avg_price)``."""
        return max(0.0, self.base_demand - self.sensitivity * avg_price)

    def shares(self, prices: np.ndarray) -> np.ndarray:
        """Softmax market share -- lower price ==> larger share.

        We feed ``-beta * price`` into a softmax.  ``beta`` controls how
        substitutable the goods are: at ``beta = 0`` everyone splits the
        market evenly regardless of price; as ``beta`` grows, the cheapest
        firm captures almost everything.
        """
        return softmax(-self.beta * np.asarray(prices, dtype=float))

    def step(self, prices: Sequence[float]) -> MarketOutcome:
        """Compute one round of the market for the given price vector."""
        prices = np.asarray(prices, dtype=float)
        avg_price = float(np.mean(prices))
        q_total = self.total_demand(avg_price)
        shares = self.shares(prices)
        demands = q_total * shares
        profits = (prices - self.costs) * demands
        return MarketOutcome(
            prices=prices,
            shares=shares,
            demands=demands,
            profits=profits,
            total_demand=q_total,
            avg_price=avg_price,
        )

    def profit_of(self, firm: int, prices: Sequence[float]) -> float:
        """Profit of a single firm for a full price vector (handy for benchmarks)."""
        return float(self.step(prices).profits[firm])

    # ------------------------------------------------------------------
    # Textbook benchmarks on the discrete grid
    # ------------------------------------------------------------------
    def best_response(self, firm: int, rival_prices: Sequence[float]) -> float:
        """Price from the menu that maximises ``firm``'s profit, rivals fixed."""
        best_price = self.price_set[0]
        best_profit = -np.inf
        for p in self.price_set:
            prices = np.asarray(rival_prices, dtype=float).copy()
            prices[firm] = p
            profit = self.profit_of(firm, prices)
            if profit > best_profit:
                best_profit = profit
                best_price = p
        return float(best_price)

    def symmetric_nash_price(self) -> float:
        """The symmetric pure-strategy Nash price on the discrete grid.

        For a symmetric market a price ``p`` is a Nash equilibrium if, when
        every firm charges ``p``, no firm can do strictly better by deviating.
        We scan the menu and return the *lowest* such price (the competitive
        one).  If no exact equilibrium exists we fall back to the price that
        minimises the incentive to deviate.
        """
        best_p = None
        best_gap = np.inf
        for p in self.price_set:
            prices = np.full(self.n_firms, p, dtype=float)
            own_profit = self.profit_of(0, prices)
            deviate_profit = -np.inf
            for q in self.price_set:
                dp = prices.copy()
                dp[0] = q
                deviate_profit = max(deviate_profit, self.profit_of(0, dp))
            gap = deviate_profit - own_profit
            if gap <= 1e-9:
                return float(p)  # exact equilibrium; lowest one wins
            if gap < best_gap:
                best_gap = gap
                best_p = p
        return float(best_p)

    def collusive_price(self) -> float:
        """Symmetric price that maximises per-firm profit (the monopoly/cartel price)."""
        best_p = self.price_set[0]
        best_profit = -np.inf
        for p in self.price_set:
            prices = np.full(self.n_firms, p, dtype=float)
            profit = self.profit_of(0, prices)
            if profit > best_profit:
                best_profit = profit
                best_p = p
        return float(best_p)

    def benchmarks(self) -> Tuple[float, float]:
        """Convenience: ``(nash_price, collusive_price)``."""
        return self.symmetric_nash_price(), self.collusive_price()
