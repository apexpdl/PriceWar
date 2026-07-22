"""
experiments.py
==============

Pre-configured PriceWar experiments plus a small command-line runner.

This is the headless entry point: it runs the simulation without Streamlit,
prints a summary, saves each run's history to ``results/`` as CSV, and emits
the rule-based explanations.  It makes the whole project runnable and
testable end-to-end without a browser::

    python experiments.py            # run all three preset experiments
    python experiments.py --exp 2    # run only experiment 2
    python experiments.py --steps 500

The three presets mirror the brief:

* **Experiment 1** -- 2 firms, identical costs (symmetric duopoly).
* **Experiment 2** -- 3 firms, different costs (asymmetric oligopoly).
* **Experiment 3** -- high competition: 4 firms, tiny cost differences and a
  high competition-intensity ``beta`` so undercutting bites hard.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from explain import explain_run
from simulation import equilibrium_summary, run_simulation, summarize
from utils import SimulationConfig


def experiment_presets() -> Dict[int, SimulationConfig]:
    """Return the three canonical experiment configurations."""
    return {
        1: SimulationConfig(
            n_firms=2,
            costs=[4.0, 4.0],
            beta=0.10,
            n_steps=1500,
            seed=42,
        ),
        2: SimulationConfig(
            n_firms=3,
            costs=[3.0, 5.0, 7.0],
            beta=0.10,
            n_steps=2000,
            seed=7,
        ),
        3: SimulationConfig(
            n_firms=4,
            costs=[4.0, 4.5, 4.0, 4.5],
            beta=0.25,          # fierce competition
            sensitivity=6.0,
            n_steps=2000,
            seed=123,
        ),
    }


EXPERIMENT_NAMES = {
    1: "Symmetric duopoly (2 firms, identical costs)",
    2: "Asymmetric oligopoly (3 firms, different costs)",
    3: "High competition (4 firms, small cost gaps, fierce rivalry)",
}


def run_experiment(exp_id: int, steps: int | None = None, save_dir: str = "results") -> None:
    """Run a single preset experiment and print its results."""
    presets = experiment_presets()
    if exp_id not in presets:
        raise ValueError(f"Unknown experiment {exp_id}; choose from {sorted(presets)}")

    config = presets[exp_id]
    if steps is not None:
        config.n_steps = steps

    print("=" * 72)
    print(f"EXPERIMENT {exp_id}: {EXPERIMENT_NAMES[exp_id]}")
    print("=" * 72)
    print(
        f"  firms={config.n_firms}  costs={config.normalized_costs()}  "
        f"beta={config.beta}  steps={config.n_steps}"
    )

    result = run_simulation(config)

    # --- Per-firm summary of the converged behaviour ---------------------
    print("\nConverged behaviour (last 200 steps):")
    print(summarize(result).to_string(index=False))

    # --- Where did the market land relative to the benchmarks? -----------
    eq = equilibrium_summary(result)
    print(
        f"\n  Nash (competitive) price : {eq['nash_price']:.1f}"
        f"\n  Collusive (cartel) price : {eq['collusive_price']:.1f}"
        f"\n  Settled average price    : {eq['mean_price']:.1f}"
        f"\n  Collusion index [0..1]   : {eq['collusion_index']:.2f}"
    )

    # --- Rule-based explanations ----------------------------------------
    print("\nAI explanation:")
    for note in explain_run(result):
        print(f"  - {note}")

    # --- Persist the raw history ----------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"experiment_{exp_id}.csv")
    result.history.to_csv(out_path, index=False)
    print(f"\nSaved full history -> {out_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PriceWar preset experiments.")
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        help="Experiment id to run (1, 2 or 3). Omit to run all three.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the number of simulation steps.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to write per-experiment CSV logs into.",
    )
    args = parser.parse_args()

    exp_ids = [args.exp] if args.exp is not None else [1, 2, 3]
    for exp_id in exp_ids:
        run_experiment(exp_id, steps=args.steps, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
