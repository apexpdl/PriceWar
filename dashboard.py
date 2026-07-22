"""
dashboard.py
============

Interactive Streamlit dashboard for PriceWar.

Run it with::

    streamlit run dashboard.py

The sidebar exposes every knob (number of firms, costs, demand shape, the
Q-learning hyper-parameters and the run length); the main pane renders the
price / profit / market-share dynamics, a convergence view, per-agent
Q-table heatmaps, the final equilibrium summary and the rule-based
"AI explanation" of what happened.

All the heavy lifting lives in the plain-Python modules (``market``,
``agent``, ``simulation``, ``explain``); this file is only presentation +
wiring, so the simulation stays fully usable without Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from explain import explain_run
from experiments import EXPERIMENT_NAMES, experiment_presets
from simulation import equilibrium_summary, run_simulation, summarize
from utils import SimulationConfig, moving_average

st.set_page_config(page_title="PriceWar — RL Market Simulator", layout="wide")

FIRM_COLORS = px.colors.qualitative.Bold


# ---------------------------------------------------------------------------
# Cached runner -- avoids re-running the whole simulation on every widget tick
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_run(config_key: tuple):
    """Run a simulation for a hashable config key and return plain data.

    Streamlit caches on the *arguments*, so we pass an immutable tuple and
    rebuild the config inside.  We return only picklable pieces (the history
    DataFrame, the equilibrium summary, explanations and the Q-tables) so the
    cache works cleanly.
    """
    config = _config_from_key(config_key)
    result = run_simulation(config)
    return {
        "history": result.history,
        "summary": summarize(result),
        "equilibrium": equilibrium_summary(result),
        "explanations": explain_run(result),
        "q_value_surfaces": [a.value_surface() for a in result.agents],
        "q_policies": [a.greedy_policy() for a in result.agents],
        "prices": list(config.prices),
        "n_firms": config.n_firms,
    }


def _config_key(config: SimulationConfig) -> tuple:
    return (
        config.n_firms,
        tuple(config.prices),
        tuple(config.normalized_costs()),
        config.base_demand,
        config.sensitivity,
        config.beta,
        config.alpha,
        config.gamma,
        config.epsilon,
        config.epsilon_min,
        config.epsilon_decay,
        config.n_steps,
        config.seed,
    )


def _config_from_key(key: tuple) -> SimulationConfig:
    (
        n_firms,
        prices,
        costs,
        base_demand,
        sensitivity,
        beta,
        alpha,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        n_steps,
        seed,
    ) = key
    return SimulationConfig(
        n_firms=n_firms,
        prices=list(prices),
        costs=list(costs),
        base_demand=base_demand,
        sensitivity=sensitivity,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        n_steps=n_steps,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def sidebar_config() -> SimulationConfig:
    st.sidebar.title("⚙️ PriceWar controls")

    preset = st.sidebar.selectbox(
        "Start from a preset",
        options=["Custom", *[f"Experiment {i}: {EXPERIMENT_NAMES[i]}" for i in (1, 2, 3)]],
        index=1,
    )

    presets = experiment_presets()
    if preset.startswith("Experiment"):
        exp_id = int(preset.split(":")[0].split()[-1])
        base = presets[exp_id]
    else:
        base = SimulationConfig()

    st.sidebar.header("Market structure")
    n_firms = st.sidebar.slider("Number of firms", 2, 4, base.n_firms)

    base_costs = base.normalized_costs()
    costs = []
    with st.sidebar.expander("Per-firm marginal cost", expanded=False):
        for i in range(n_firms):
            default = base_costs[i] if i < len(base_costs) else base_costs[-1]
            costs.append(
                st.slider(f"Firm {i} cost", 0.0, 20.0, float(default), 0.5, key=f"cost_{i}")
            )

    st.sidebar.header("Demand model")
    base_demand = st.sidebar.slider("Base market size (demand intercept)", 50.0, 400.0, base.base_demand, 10.0)
    sensitivity = st.sidebar.slider("Price sensitivity (demand slope)", 1.0, 12.0, base.sensitivity, 0.5)
    beta = st.sidebar.slider("Competition intensity (β)", 0.0, 0.6, base.beta, 0.01,
                             help="Higher β → customers punish price differences harder → fiercer competition.")

    st.sidebar.header("Q-learning")
    alpha = st.sidebar.slider("Learning rate (α)", 0.01, 0.9, base.alpha, 0.01)
    gamma = st.sidebar.slider("Discount factor (γ)", 0.0, 0.99, base.gamma, 0.01)
    epsilon = st.sidebar.slider("Initial exploration (ε)", 0.0, 1.0, base.epsilon, 0.05)
    epsilon_min = st.sidebar.slider("Min exploration (ε floor)", 0.0, 0.2, base.epsilon_min, 0.01)
    auto_decay = st.sidebar.checkbox("Auto-tune ε decay to run length", value=True)
    epsilon_decay = None
    if not auto_decay:
        epsilon_decay = st.sidebar.slider("ε decay per step", 0.990, 0.9999, 0.999, 0.0001, format="%.4f")

    st.sidebar.header("Simulation")
    n_steps = st.sidebar.slider("Number of steps", 200, 4000, base.n_steps, 100)
    seed = st.sidebar.number_input("Random seed", value=int(base.seed), step=1)

    return SimulationConfig(
        n_firms=n_firms,
        prices=list(base.prices),
        costs=costs,
        base_demand=base_demand,
        sensitivity=sensitivity,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        n_steps=n_steps,
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def line_by_firm(history: pd.DataFrame, y: str, title: str, y_label: str, smooth: int) -> go.Figure:
    fig = go.Figure()
    for firm, grp in history.groupby("firm"):
        grp = grp.sort_values("step")
        y_vals = moving_average(grp[y].to_numpy(), smooth) if smooth > 1 else grp[y].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=grp["step"],
                y=y_vals,
                mode="lines",
                name=f"Firm {int(firm)}",
                line=dict(color=FIRM_COLORS[int(firm) % len(FIRM_COLORS)]),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Simulation step",
        yaxis_title=y_label,
        legend_title="",
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
    )
    return fig


def convergence_chart(history: pd.DataFrame, window: int = 100) -> go.Figure:
    """Rolling price dispersion across firms -- how close to 'settled' we are."""
    per_step = history.groupby("step")["price"].std(ddof=0).fillna(0.0)
    smoothed = moving_average(per_step.to_numpy(), window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=per_step.index, y=per_step.to_numpy(), mode="lines",
                             name="cross-firm price spread", line=dict(color="#9aa0a6", width=1)))
    fig.add_trace(go.Scatter(x=per_step.index, y=smoothed, mode="lines",
                             name=f"{window}-step average", line=dict(color="#1a73e8", width=3)))
    fig.update_layout(
        title="Convergence — dispersion of prices across firms (lower = more settled)",
        xaxis_title="Simulation step",
        yaxis_title="Std. dev. of prices",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    return fig


def q_heatmap(surface: np.ndarray, prices, firm: int) -> go.Figure:
    fig = px.imshow(
        surface,
        labels=dict(x="Competitor avg price", y="Own previous price", color="Max Q-value"),
        x=[f"{p:.0f}" for p in prices],
        y=[f"{p:.0f}" for p in prices],
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_layout(title=f"Firm {firm}: learned value by state", height=320,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def policy_heatmap(policy: np.ndarray, prices, firm: int) -> go.Figure:
    price_grid = np.asarray(prices)[policy]
    fig = px.imshow(
        price_grid,
        labels=dict(x="Competitor avg price", y="Own previous price", color="Chosen price"),
        x=[f"{p:.0f}" for p in prices],
        y=[f"{p:.0f}" for p in prices],
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(title=f"Firm {firm}: greedy pricing policy", height=320,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🥇 PriceWar — Reinforcement Learning Market Simulator")
    st.caption(
        "Independent Q-learning firms set prices each round. Watch them compete — "
        "and sometimes tacitly coordinate on high prices — without ever communicating."
    )

    config = sidebar_config()

    run = st.sidebar.button("▶️ Run simulation", type="primary", use_container_width=True)
    if run or "last_key" not in st.session_state:
        st.session_state["last_key"] = _config_key(config)

    key = st.session_state["last_key"]
    with st.spinner("Training pricing agents…"):
        data = _cached_run(key)

    history = data["history"]
    eq = data["equilibrium"]

    # --- Top-line equilibrium metrics ------------------------------------
    st.subheader("Final equilibrium summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Settled avg price", f"{eq['mean_price']:.1f}")
    c2.metric("Nash (competitive)", f"{eq['nash_price']:.1f}")
    c3.metric("Collusive (cartel)", f"{eq['collusive_price']:.1f}")
    c4.metric("Collusion index", f"{eq['collusion_index']:.0%}",
              help="0% = fully competitive (Nash), 100% = fully collusive (monopoly price).")
    c5.metric("Avg profit / firm", f"{eq['mean_profit']:.0f}")

    st.progress(eq["collusion_index"], text="Competition ⟵ collusion index ⟶ Collusion")

    smooth = st.slider("Chart smoothing (moving-average window)", 1, 200, 25)

    # --- Dynamics --------------------------------------------------------
    st.subheader("Market dynamics")
    left, right = st.columns(2)
    left.plotly_chart(line_by_firm(history, "price", "Price over time", "Price", smooth),
                      use_container_width=True)
    right.plotly_chart(line_by_firm(history, "profit", "Profit over time", "Profit", smooth),
                       use_container_width=True)

    left2, right2 = st.columns(2)
    left2.plotly_chart(line_by_firm(history, "market_share", "Market share over time", "Share", smooth),
                       use_container_width=True)
    right2.plotly_chart(convergence_chart(history), use_container_width=True)

    # --- AI explanation --------------------------------------------------
    st.subheader("🤖 AI explanation (rule-based)")
    for note in data["explanations"]:
        st.markdown(f"- {note}")

    # --- Per-firm summary table -----------------------------------------
    st.subheader("Per-firm converged behaviour")
    st.dataframe(data["summary"], use_container_width=True, hide_index=True)

    # --- Q-tables --------------------------------------------------------
    st.subheader("Learned Q-tables")
    st.caption(
        "Left: how valuable each state looks to the firm. Right: the price the firm would "
        "pick in each state once it stops exploring."
    )
    firm_tabs = st.tabs([f"Firm {i}" for i in range(data["n_firms"])])
    for i, tab in enumerate(firm_tabs):
        with tab:
            hcol, pcol = st.columns(2)
            hcol.plotly_chart(q_heatmap(data["q_value_surfaces"][i], data["prices"], i),
                              use_container_width=True)
            pcol.plotly_chart(policy_heatmap(data["q_policies"][i], data["prices"], i),
                              use_container_width=True)

    # --- Step inspector --------------------------------------------------
    with st.expander("🔍 Inspect a single step"):
        step = st.slider("Step", 0, int(history["step"].max()), int(history["step"].max()))
        row = history[history["step"] == step]
        st.dataframe(row[["firm", "price", "demand", "market_share", "profit", "epsilon"]],
                     use_container_width=True, hide_index=True)
        # Reconstruct a lightweight result-like object for explain_step.
        st.text(_explain_step_from_history(history, step))

    st.download_button(
        "⬇️ Download full history (CSV)",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="pricewar_history.csv",
        mime="text/csv",
    )


def _explain_step_from_history(history: pd.DataFrame, step: int) -> str:
    """Standalone version of explain_step that works off just the history frame."""
    row = history[history["step"] == step]
    if row.empty:
        return "No data for this step."
    prev = history[history["step"] == step - 1] if step > 0 else None
    parts = [
        f"Step {step}: average price {row['avg_price'].iloc[0]:.1f}, "
        f"total market demand {row['total_demand'].iloc[0]:.1f}."
    ]
    for _, r in row.iterrows():
        firm = int(r["firm"])
        line = (f"Firm {firm}: price {r['price']:.0f}, share {r['market_share']*100:.0f}%, "
                f"profit {r['profit']:.1f}")
        if prev is not None and not prev.empty:
            prev_price = prev[prev["firm"] == firm]["price"]
            if not prev_price.empty:
                delta = r["price"] - prev_price.iloc[0]
                line += (f" (raised price by {delta:.0f})" if delta > 0
                         else f" (cut price by {abs(delta):.0f})" if delta < 0
                         else " (held price)")
        parts.append(line)
    return "\n".join(parts)


if __name__ == "__main__":
    main()
