"""
simulation.py
=============

Glue that runs the repeated pricing game: build a :class:`market.Market`,
create one :class:`agent.QLearningAgent` per firm, then loop for
``n_steps``.  Every round is recorded so the dashboard and the explanation
layer have a complete, tidy history to work from.

The loop is deliberately synchronous and easy to read:

    for each step:
        1. each agent observes the current state
        2. each agent picks a price (epsilon-greedy)
        3. the market turns prices into demand + profit
        4. each agent updates its Q-table using its own profit as reward
        5. everything is logged
        6. epsilon decays

The "state" an agent sees is built from *last* round's prices, so the agents
are genuinely reacting to one another over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from agent import QLearningAgent
from market import Market
from utils import SimulationConfig


@dataclass
class SimulationResult:
    """Everything a caller needs after a run."""

    history: pd.DataFrame            #: long-format per-firm per-step log
    agents: List[QLearningAgent]     #: trained agents (Q-tables inside)
    market: Market                   #: the environment (knows benchmarks)
    config: SimulationConfig         #: the config that produced this run


class Simulation:
    """Owns the market, the agents and the main learning loop."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        costs = config.normalized_costs()
        self.market = Market(
            prices=config.prices,
            costs=costs,
            base_demand=config.base_demand,
            sensitivity=config.sensitivity,
            beta=config.beta,
        )

        self.agents: List[QLearningAgent] = [
            QLearningAgent(
                firm_id=i,
                price_set=config.prices,
                alpha=config.alpha,
                gamma=config.gamma,
                epsilon=config.epsilon,
                epsilon_min=config.epsilon_min,
                epsilon_decay=config.effective_epsilon_decay(),
                # Give each agent its own RNG stream for reproducibility.
                rng=np.random.default_rng(config.seed + 1 + i),
            )
            for i in range(config.n_firms)
        ]

        # Kick things off with a random price per firm so the first state is
        # well-defined.
        self.prices = self.rng.choice(config.prices, size=config.n_firms)

    # ------------------------------------------------------------------
    def _competitor_avg(self, prices: np.ndarray, firm: int) -> float:
        """Average price of *everyone except* ``firm``."""
        if len(prices) == 1:
            return float(prices[firm])  # monopoly: own price is all there is
        mask = np.ones(len(prices), dtype=bool)
        mask[firm] = False
        return float(np.mean(prices[mask]))

    # ------------------------------------------------------------------
    def run(self, progress_callback=None) -> SimulationResult:
        """Run the full simulation and return a :class:`SimulationResult`.

        ``progress_callback`` (optional) is called as ``callback(step, n_steps)``
        roughly 100 times over the run -- the Streamlit dashboard uses it to
        drive a progress bar.
        """
        cfg = self.config
        prices = np.asarray(self.prices, dtype=float).copy()
        records = []

        report_every = max(1, cfg.n_steps // 100)

        for step in range(cfg.n_steps):
            # 1. Observe current state (built from last round's prices).
            states = []
            for i, agent in enumerate(self.agents):
                comp_avg = self._competitor_avg(prices, i)
                states.append(agent.encode_state(prices[i], comp_avg))

            # 2. Choose actions -> new prices.
            actions = [agent.select_action(states[i]) for i, agent in enumerate(self.agents)]
            new_prices = np.array(
                [agent.price_for_action(actions[i]) for i, agent in enumerate(self.agents)],
                dtype=float,
            )

            # 3. Market resolves the round.
            outcome = self.market.step(new_prices)

            # 4. Learn: reward is each firm's own profit; next state uses the
            #    prices that were just played.
            for i, agent in enumerate(self.agents):
                comp_avg_next = self._competitor_avg(new_prices, i)
                next_state = agent.encode_state(new_prices[i], comp_avg_next)
                agent.update(states[i], actions[i], outcome.profits[i], next_state)

            # 5. Log one row per firm.
            for i in range(cfg.n_firms):
                records.append(
                    {
                        "step": step,
                        "firm": i,
                        "price": float(new_prices[i]),
                        "demand": float(outcome.demands[i]),
                        "profit": float(outcome.profits[i]),
                        "market_share": float(outcome.shares[i]),
                        "cost": float(self.market.costs[i]),
                        "epsilon": float(self.agents[i].epsilon),
                        "total_demand": float(outcome.total_demand),
                        "avg_price": float(outcome.avg_price),
                    }
                )

            # 6. Decay exploration and roll prices forward.
            for agent in self.agents:
                agent.decay_epsilon()
            prices = new_prices

            if progress_callback is not None and step % report_every == 0:
                progress_callback(step, cfg.n_steps)

        if progress_callback is not None:
            progress_callback(cfg.n_steps, cfg.n_steps)

        history = pd.DataFrame.from_records(records)
        return SimulationResult(
            history=history,
            agents=self.agents,
            market=self.market,
            config=cfg,
        )


def run_simulation(config: SimulationConfig, progress_callback=None) -> SimulationResult:
    """Convenience wrapper: build a :class:`Simulation` and run it."""
    return Simulation(config).run(progress_callback=progress_callback)


# ---------------------------------------------------------------------------
# Post-run summaries (used by the dashboard and the CLI experiments)
# ---------------------------------------------------------------------------
def summarize(result: SimulationResult, tail: int = 200) -> pd.DataFrame:
    """Per-firm summary of the *converged* behaviour (last ``tail`` steps)."""
    hist = result.history
    last = hist[hist["step"] >= hist["step"].max() - tail + 1]
    rows = []
    for firm, grp in last.groupby("firm"):
        rows.append(
            {
                "firm": int(firm),
                "cost": float(grp["cost"].iloc[0]),
                "avg_price": float(grp["price"].mean()),
                "avg_profit": float(grp["profit"].mean()),
                "avg_market_share": float(grp["market_share"].mean()),
                "price_std": float(grp["price"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def equilibrium_summary(result: SimulationResult, tail: int = 200) -> dict:
    """High-level facts about where the market settled."""
    hist = result.history
    last = hist[hist["step"] >= hist["step"].max() - tail + 1]
    nash, collusive = result.market.benchmarks()
    mean_price = float(last["price"].mean())

    # Where does the settled price sit between the competitive Nash price and
    # the fully-collusive monopoly price?  1.0 = full collusion, 0.0 = Nash.
    if collusive > nash:
        collusion_index = (mean_price - nash) / (collusive - nash)
    else:
        collusion_index = 0.0
    collusion_index = float(np.clip(collusion_index, 0.0, 1.0))

    return {
        "mean_price": mean_price,
        "mean_profit": float(last["profit"].mean()),
        "total_profit": float(last.groupby("step")["profit"].sum().mean()),
        "nash_price": nash,
        "collusive_price": collusive,
        "collusion_index": collusion_index,
        "price_stability": float(last.groupby("firm")["price"].std(ddof=0).mean()),
    }
