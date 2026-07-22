# 🥇 PriceWar — Reinforcement Learning Market Simulator

**PriceWar simulates how independent AI pricing agents learn to compete — or unintentionally
coordinate — without ever communicating.**

Each firm is a tabular **Q-learning** agent that sets a price every round. Demand and profit
follow a simple, transparent economic model. Over hundreds of rounds you can watch prices
evolve into competition, stabilisation, oscillation (price cycles), or *tacit collusion* on
high prices — the phenomenon economists and antitrust regulators are actively researching.

> This is an **educational** simulation for building economics + ML intuition, not a
> production ML system. It uses **NumPy + Pandas + Streamlit + Plotly** and *no* deep-learning
> frameworks.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Run the interactive dashboard
streamlit run dashboard.py

# 2b. …or run the preset experiments headlessly (no browser needed)
python experiments.py            # runs all three presets, saves CSVs to results/
python experiments.py --exp 2    # run just one
python experiments.py --steps 800

# 3. Run the tests
python test_pricewar.py          # or: pytest test_pricewar.py
```

---

## What you'll see

The dashboard (`streamlit run dashboard.py`) gives you:

| Control (sidebar) | Output (main pane) |
|---|---|
| Number of firms (2–4) | Price over time (per firm) |
| Per-firm marginal cost | Profit over time |
| Demand size & price sensitivity | Market share over time |
| Competition intensity **β** | Convergence view (price dispersion) |
| Learning rate **α**, discount **γ** | Per-agent **Q-table heatmaps** & greedy policy |
| Exploration **ε** (+ decay) | **Final equilibrium summary** vs. Nash & cartel prices |
| Number of steps, random seed | 🤖 **Rule-based "AI explanation"** of what happened |

The headline number is the **collusion index** (0% = fully competitive Nash pricing,
100% = fully collusive monopoly pricing), computed by comparing where prices settled against
two benchmarks PriceWar solves for on the discrete price grid.

---

## Default experiments

| # | Setup | Typical result |
|---|---|---|
| **1** | 2 firms, identical costs | Tacit coordination — prices drift above competitive level |
| **2** | 3 firms, different costs | Low-cost firm anchors the market; still coordinates |
| **3** | 4 firms, tiny cost gaps, high β | Fierce competition — prices pushed toward the Nash floor |

Run them with `python experiments.py`.

---

## Project layout

```
market.py            # Economic environment: demand, share, profit + Nash/cartel benchmarks
agent.py             # Tabular Q-learning agent (state, ε-greedy action, Q-update)
simulation.py        # The repeated-game loop + Pandas logging + summaries
explain.py           # Rule-based "AI explanation" of a finished run
experiments.py       # Three presets + headless CLI runner
dashboard.py         # Streamlit + Plotly interactive UI
utils.py             # SimulationConfig + small numeric helpers
test_pricewar.py     # End-to-end sanity tests
PROJECT_DOCUMENTATION.md   # Full write-up (model, RL, economics, interview notes)
WHY.md               # Every engineering decision, alternatives, and trade-offs
```

## How it works, in one paragraph

Every round, each firm observes a small discrete **state** — its own last price and the
average competitor price, each bucketed to the price menu — and picks a price with an
**ε-greedy** policy. The **market** turns the price vector into total demand (which falls as
the average price rises) and market shares (a softmax that rewards undercutting), then into
profit `= (price − cost) × quantity`. Each firm treats its **own profit** as the reward and
applies the standard **Q-learning update**. Exploration decays over time, so the agents
gradually lock in a policy — and because they keep re-encountering one another, that policy is
shaped by their rivals' behaviour, which is where the emergent competition or coordination
comes from.


