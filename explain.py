"""
explain.py
==========

A lightweight, **rule-based** "AI explanation" layer.

There is no LLM here -- and there doesn't need to be.  The interesting
behaviour in PriceWar (undercutting, stabilisation, oscillation, tacit
coordination) shows up as simple, detectable patterns in the logged history.
This module reads that history and turns the patterns into plain-English
sentences a student or economist can immediately understand.

Everything is a pure function of the :class:`simulation.SimulationResult`,
so the same explanations can be printed on the command line or rendered in
the Streamlit dashboard.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from simulation import SimulationResult, equilibrium_summary
from utils import moving_average


def _trend(series: np.ndarray, window: int = 200) -> float:
    """Signed slope of a series over its final ``window`` points.

    Positive = rising, negative = falling.  Uses a plain least-squares fit so
    a single noisy point doesn't flip the verdict.
    """
    tail = np.asarray(series, dtype=float)[-window:]
    if tail.size < 2:
        return 0.0
    x = np.arange(tail.size)
    slope = np.polyfit(x, tail, 1)[0]
    return float(slope)


def explain_run(result: SimulationResult, tail: int = 200) -> List[str]:
    """Return a list of human-readable observations about a finished run."""
    hist = result.history
    cfg = result.config
    summary = equilibrium_summary(result, tail=tail)
    notes: List[str] = []

    nash = summary["nash_price"]
    collusive = summary["collusive_price"]
    mean_price = summary["mean_price"]
    ci = summary["collusion_index"]

    # --- 1. Overall regime: competitive vs. coordinated -------------------
    if ci >= 0.66:
        notes.append(
            f"The market settled at an average price of {mean_price:.1f}, close to the "
            f"joint-profit-maximising ('collusive') price of {collusive:.1f}. Even though "
            f"the {cfg.n_firms} firms never communicate, their independent Q-learning "
            f"policies converged on high, mutually profitable prices -- an example of "
            f"tacit coordination emerging from repeated interaction."
        )
    elif ci <= 0.33:
        notes.append(
            f"The market settled near the competitive Nash price of {nash:.1f} "
            f"(average price {mean_price:.1f}). Firms kept undercutting one another to win "
            f"market share, so prices stayed low and profits thin -- the classic Bertrand "
            f"competition outcome."
        )
    else:
        notes.append(
            f"The market settled at an average price of {mean_price:.1f}, between the "
            f"competitive Nash level ({nash:.1f}) and the collusive level ({collusive:.1f}). "
            f"Firms are partially coordinating: prices are above pure competition but below "
            f"full monopoly pricing."
        )

    # --- 2. Convergence vs. oscillation -----------------------------------
    stability = summary["price_stability"]
    price_span = max(cfg.prices) - min(cfg.prices)
    if stability < 0.05 * price_span:
        notes.append(
            "Prices have effectively stabilised: over the final stretch each firm "
            "repeatedly plays the same price, meaning the agents' reward expectations have "
            "equalised and no firm sees a reason to deviate."
        )
    elif stability > 0.20 * price_span:
        notes.append(
            "Prices are still oscillating rather than settling. This usually reflects an "
            "'edgeworth cycle': a firm undercuts to grab share, rivals follow it down, then "
            "someone resets to a high price and the cycle repeats."
        )
    else:
        notes.append(
            "Prices are mostly stable with occasional exploratory moves as residual "
            "epsilon-greedy exploration nudges firms to re-test alternative prices."
        )

    # --- 3. Per-firm colour: who moved and why ----------------------------
    for firm, grp in hist.groupby("firm"):
        grp = grp.sort_values("step")
        price_trend = _trend(grp["price"].to_numpy())
        profit_now = grp["profit"].tail(tail).mean()
        profit_before = grp["profit"].head(len(grp) // 2).mean()
        cost = grp["cost"].iloc[0]

        if price_trend > 0.002 * price_span:
            direction = "raised its prices over time"
            reason = (
                "learning that higher prices earned more profit once rivals stopped "
                "aggressively undercutting"
                if profit_now >= profit_before
                else "chasing rivals upward, though its own profit did not clearly improve"
            )
        elif price_trend < -0.002 * price_span:
            direction = "lowered its prices over time"
            reason = (
                "using undercutting to capture additional market share"
            )
        else:
            direction = "held its price roughly constant"
            reason = "having found a price whose expected reward it could not beat by deviating"

        notes.append(
            f"Firm {int(firm)} (cost {cost:.1f}) {direction}, {reason}. "
            f"Its recent average profit is {profit_now:.1f}."
        )

    # --- 4. Cost asymmetry note ------------------------------------------
    costs = result.market.costs
    if float(np.std(costs)) > 1e-6:
        cheapest = int(np.argmin(costs))
        priciest_firm_price = (
            hist[hist["firm"] == cheapest]["price"].tail(tail).mean()
        )
        notes.append(
            f"Firm {cheapest} has the lowest cost ({costs[cheapest]:.1f}), which gives it "
            f"the most room to undercut; it tends to anchor the market's price level "
            f"(recent average price {priciest_firm_price:.1f})."
        )

    return notes


def explain_step(result: SimulationResult, step: int) -> str:
    """Explain what happened at one particular ``step`` (used by the dashboard slider)."""
    hist = result.history
    row = hist[hist["step"] == step]
    if row.empty:
        return "No data for this step."

    prev = hist[hist["step"] == step - 1] if step > 0 else None
    parts: List[str] = [f"Step {step}: average price {row['avg_price'].iloc[0]:.1f}, "
                        f"total market demand {row['total_demand'].iloc[0]:.1f}."]

    for _, r in row.iterrows():
        firm = int(r["firm"])
        line = (
            f"Firm {firm}: price {r['price']:.0f}, share {r['market_share']*100:.0f}%, "
            f"profit {r['profit']:.1f}"
        )
        if prev is not None and not prev.empty:
            prev_price = prev[prev["firm"] == firm]["price"]
            if not prev_price.empty:
                delta = r["price"] - prev_price.iloc[0]
                if delta > 0:
                    line += f" (raised price by {delta:.0f})"
                elif delta < 0:
                    line += f" (cut price by {abs(delta):.0f})"
                else:
                    line += " (held price)"
        parts.append(line)

    return "\n".join(parts)
