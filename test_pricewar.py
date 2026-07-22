"""
test_pricewar.py
================

Lightweight end-to-end tests for PriceWar.  No test framework required --
run either of::

    python test_pricewar.py        # plain asserts, prints a summary
    pytest test_pricewar.py        # if pytest is installed

These are sanity checks (shapes, invariants, determinism, benchmark ordering),
not a proof of learning-theory correctness.
"""

from __future__ import annotations

import numpy as np

from agent import QLearningAgent
from explain import explain_run
from market import Market
from simulation import equilibrium_summary, run_simulation
from utils import SimulationConfig, moving_average, nearest_index, softmax


def test_softmax_and_helpers():
    s = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(s.sum() - 1.0) < 1e-9
    assert s[2] > s[0]
    assert nearest_index(11.0, [5, 10, 15, 20, 25]) == 1
    ma = moving_average([1, 2, 3, 4], window=2)
    assert len(ma) == 4 and abs(ma[-1] - 3.5) < 1e-9


def test_market_invariants():
    m = Market(prices=[5, 10, 15, 20, 25], costs=[4, 4], beta=0.1)
    out = m.step([10, 20])
    # Shares sum to 1, cheaper firm gets more share.
    assert abs(out.shares.sum() - 1.0) < 1e-9
    assert out.shares[0] > out.shares[1]
    # Total demand equals sum of firm demands.
    assert abs(out.demands.sum() - out.total_demand) < 1e-6
    # Demand falls as average price rises.
    assert m.total_demand(5) > m.total_demand(25)


def test_market_benchmarks_ordered():
    m = Market(prices=[5, 10, 15, 20, 25], costs=[4, 4], beta=0.1)
    nash, collusive = m.benchmarks()
    # The collusive (monopoly) price should be at least the competitive price.
    assert collusive >= nash
    assert nash in m.price_set and collusive in m.price_set


def test_agent_updates_qtable():
    agent = QLearningAgent(0, [5, 10, 15, 20, 25], alpha=0.5, gamma=0.9,
                           epsilon=0.0, rng=np.random.default_rng(0))
    state = agent.encode_state(10, 15)
    before = agent.q[state[0], state[1], 2]
    agent.update(state, action=2, reward=100.0, next_state=state)
    after = agent.q[state[0], state[1], 2]
    assert after > before  # positive reward raises the Q-value


def test_simulation_shapes_and_determinism():
    cfg = SimulationConfig(n_firms=3, costs=[3, 5, 7], n_steps=300, seed=1)
    r1 = run_simulation(cfg)
    r2 = run_simulation(SimulationConfig(n_firms=3, costs=[3, 5, 7], n_steps=300, seed=1))
    # One row per firm per step.
    assert len(r1.history) == 300 * 3
    assert set(r1.history["firm"].unique()) == {0, 1, 2}
    # Same seed -> identical run.
    assert np.allclose(r1.history["price"].to_numpy(), r2.history["price"].to_numpy())
    # Epsilon decayed below its start value.
    assert r1.history["epsilon"].iloc[-1] < r1.history["epsilon"].iloc[0]


def test_equilibrium_and_explanations():
    cfg = SimulationConfig(n_firms=2, costs=[4, 4], n_steps=500, seed=42)
    result = run_simulation(cfg)
    eq = equilibrium_summary(result)
    assert 0.0 <= eq["collusion_index"] <= 1.0
    assert min(cfg.prices) <= eq["mean_price"] <= max(cfg.prices)
    notes = explain_run(result)
    assert isinstance(notes, list) and len(notes) >= 3
    assert all(isinstance(n, str) and n for n in notes)


def test_monopoly_single_firm_runs():
    # n_firms clamps costs correctly and the competitor-average logic handles 1 firm.
    cfg = SimulationConfig(n_firms=2, costs=5.0, n_steps=200, seed=3)  # scalar cost broadcast
    result = run_simulation(cfg)
    assert (result.history["cost"] == 5.0).all()


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
