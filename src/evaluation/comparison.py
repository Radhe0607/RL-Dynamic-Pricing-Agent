"""
comparison.py
-------------
Baseline vs DQN comparison harness for the Dynamic Pricing Agent.

This module provides a unified pipeline that runs every pricing strategy
— three hand-crafted baselines plus the trained DQN agent — over the
same set of episodes and computes a consistent set of KPIs for each.

KPIs computed per strategy
---------------------------
  total_revenue     : Sum of (price × units_sold) across all episodes.
  total_bookings    : Total units sold (demand fulfilled) across all episodes.
  occupancy_rate    : total_bookings / (episodes × initial_inventory) —
                      fraction of available inventory sold.
  avg_selling_price : Weighted mean price across all selling events.
  avg_reward        : Mean episode reward (directly from env.step).
  mean_reward       : Alias for avg_reward (included for interoperability).
  std_reward        : Standard deviation of episode rewards.

How comparison results are calculated
--------------------------------------
Each strategy is evaluated using the same PricingEnvironment.
Bookings, realized price, revenue, and reward come from env.step(),
so results respect the simulator's demand noise and inventory limit.
Occupancy rate measures the share of each episode's initial inventory sold.

Outputs
--------
Results are saved to ``outputs/`` in two formats:
  - ``outputs/comparison_results.csv``  — tabular, one row per strategy
  - ``outputs/comparison_results.json`` — machine-readable, full detail

Public API
----------
    run_comparison(checkpoint_path, n_episodes, cfg, output_dir) → dict
        Main entry point.  Runs all strategies and saves results.

    compare_strategies(strategies, n_episodes, max_steps) → dict
        Lower-level runner; accepts any list of strategy objects.

Usage
-----
    from src.evaluation.comparison import run_comparison

    results = run_comparison(
        checkpoint_path="checkpoints/dqn_final.pt",
        n_episodes=20,
    )
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.baselines.strategies import (
    BaseStrategy,
    FixedPriceStrategy,
    RandomPriceStrategy,
    RuleBasedPricingStrategy,
)
from src.config.dqn_config import DQNConfig
from src.environment.pricing_env import PricingEnvironment


# ---------------------------------------------------------------------------
# DQN adapter — wraps DQNAgent to match the BaseStrategy interface
# ---------------------------------------------------------------------------

class _DQNStrategyAdapter(BaseStrategy):
    """Thin adapter that makes a trained ``DQNAgent`` look like a
    ``BaseStrategy`` so it can be passed to ``compare_strategies``.

    Args:
        agent     : A loaded ``DQNAgent`` instance.
        n_actions (int): Action space size.
    """

    def __init__(self, agent: Any, n_actions: int = 5) -> None:
        super().__init__(n_actions=n_actions, name="DQN Agent")
        self._agent = agent

    def select_action(self, state: np.ndarray) -> int:
        """Delegate to the agent's greedy policy (ε = 0).

        Args:
            state (np.ndarray): Current observation.

        Returns:
            int: Greedy action chosen by the DQN.
        """
        return self._agent.select_action(state, epsilon=0.0)


# ---------------------------------------------------------------------------
# Internal episode runner
# ---------------------------------------------------------------------------

def _run_strategy_episode(
    strategy:    BaseStrategy,
    env:         PricingEnvironment,
    max_steps:   int,
) -> Dict[str, float]:
    """Roll out one full episode for a given strategy and collect KPI signals.

    Reward and pricing KPIs are all taken from the same environment step,
    keeping realized sales, inventory limits, revenue, and reward consistent.

    Args:
        strategy  (BaseStrategy):        The strategy to evaluate.
        env       (PricingEnvironment):  Gymnasium environment instance.
        max_steps (int):                 Hard cap on episode length.

    Returns:
        dict: Per-episode KPI signals:
              ``total_reward``, ``total_revenue``, ``total_bookings``,
              ``steps``, ``prices_used``.
    """
    state, _ = env.reset()
    state    = np.array(state, dtype=np.float32)
    strategy.reset()

    total_reward   = 0.0
    total_revenue  = 0.0
    total_bookings = 0
    steps          = 0

    for _ in range(max_steps):
        action = strategy.select_action(state)

        # Use the environment's realized economics for every strategy.
        next_state, step_reward, terminated, truncated, info = env.step(action)
        bookings = int(info["units_sold"])
        revenue = float(info["revenue"])

        total_reward   += step_reward
        total_revenue  += revenue
        total_bookings += bookings

        steps += 1
        state  = np.array(next_state, dtype=np.float32)

        if terminated or truncated:
            break

    return {
        "total_reward":   total_reward,
        "total_revenue":  total_revenue,
        "total_bookings": total_bookings,
        "steps":          steps,
    }


# ---------------------------------------------------------------------------
# KPI aggregation
# ---------------------------------------------------------------------------

def _aggregate_kpis(
    episode_results: List[Dict[str, float]],
    inventory_per_episode: int,
) -> Dict[str, float]:
    """Aggregate per-episode signals into strategy-level KPIs.

    Args:
        episode_results (list[dict]): Output of ``_run_strategy_episode``
                                      for each episode.
        inventory_per_episode (int): Initial units available in each episode.

    Returns:
        dict: Strategy-level KPIs with the following keys:
              ``total_revenue``, ``total_bookings``, ``occupancy_rate``,
              ``avg_selling_price``, ``avg_reward``, ``mean_reward``,
              ``std_reward``.
    """
    n = len(episode_results)

    rewards   = [r["total_reward"]   for r in episode_results]
    revenues  = [r["total_revenue"]  for r in episode_results]
    bookings  = [r["total_bookings"] for r in episode_results]
    total_revenue   = sum(revenues)
    total_bookings  = sum(bookings)
    total_capacity  = n * inventory_per_episode
    occupancy_rate  = total_bookings / max(total_capacity, 1)
    avg_sell_price  = total_revenue / total_bookings if total_bookings else 0.0
    avg_reward      = float(np.mean(rewards))
    std_reward      = float(np.std(rewards)) if n > 1 else 0.0

    return {
        "total_revenue":    round(total_revenue,  4),
        "total_bookings":   total_bookings,
        "occupancy_rate":   round(occupancy_rate,  4),
        "avg_selling_price": round(avg_sell_price, 4),
        "avg_reward":       round(avg_reward,      4),
        "mean_reward":      round(avg_reward,      4),   # alias
        "std_reward":       round(std_reward,      4),
    }


# ---------------------------------------------------------------------------
# Core comparison runner
# ---------------------------------------------------------------------------

def compare_strategies(
    strategies:  List[BaseStrategy],
    n_episodes:  int = 20,
    max_steps:   int = 200,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a list of strategies and return their KPIs.

    All strategies use the same environment dynamics; demand noise is sampled
    within each rollout according to the environment's seeded RNG.

    Args:
        strategies  (list[BaseStrategy]): Strategies to evaluate.
        n_episodes  (int):                Episodes per strategy. Default 20.
        max_steps   (int):                Steps per episode.    Default 200.

    Returns:
        dict: ``{strategy_name: kpi_dict}`` — one entry per strategy.
    """
    env     = PricingEnvironment()
    results = {}

    for strategy in strategies:
        print(f"  [compare] Running '{strategy.name}' for {n_episodes} episodes …")
        episode_data = [
            _run_strategy_episode(strategy, env, max_steps)
            for _ in range(n_episodes)
        ]
        kpis = _aggregate_kpis(episode_data, env.initial_inventory)
        results[strategy.name] = kpis

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(results: Dict[str, Dict], filepath: str) -> None:
    """Write the comparison results to a CSV file.

    Columns: strategy_name, total_revenue, total_bookings, occupancy_rate,
             avg_selling_price, avg_reward, std_reward.

    Args:
        results  (dict):  Output of ``compare_strategies``.
        filepath (str):   Destination CSV path.
    """
    kpi_fields = [
        "total_revenue", "total_bookings", "occupancy_rate",
        "avg_selling_price", "avg_reward", "std_reward",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["strategy"] + kpi_fields)
        writer.writeheader()
        for strategy_name, kpis in results.items():
            row = {"strategy": strategy_name}
            row.update({k: kpis.get(k, "") for k in kpi_fields})
            writer.writerow(row)

    print(f"  [compare] CSV  saved → {filepath}")


def _save_json(
    results:       Dict[str, Dict],
    filepath:      str,
    n_episodes:    int,
    max_steps:     int,
    checkpoint:    Optional[str],
) -> None:
    """Write the comparison results to a JSON file with metadata.

    Args:
        results    (dict):  Output of ``compare_strategies``.
        filepath   (str):   Destination JSON path.
        n_episodes (int):   Episodes run per strategy.
        max_steps  (int):   Steps per episode.
        checkpoint (str|None): Checkpoint path used for the DQN agent.
    """
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_episodes":   n_episodes,
        "max_steps":    max_steps,
        "dqn_checkpoint": checkpoint,
        "strategies":   results,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  [compare] JSON saved → {filepath}")


def _print_comparison_table(results: Dict[str, Dict]) -> None:
    """Print a formatted side-by-side comparison table to stdout.

    Args:
        results (dict): Output of ``compare_strategies``.
    """
    sep = "=" * 90
    col = 18   # column width

    print(sep)
    print("  Strategy Comparison Results")
    print(sep)

    # Header
    header = f"  {'Strategy':<28}"
    for kpi in ["Revenue", "Bookings", "Occupancy%", "Avg Price", "Avg Reward"]:
        header += f"  {kpi:>{col}}"
    print(header)
    print("  " + "-" * 86)

    for name, kpis in results.items():
        row = f"  {name:<28}"
        row += f"  {kpis['total_revenue']:>{col}.2f}"
        row += f"  {kpis['total_bookings']:>{col}}"
        row += f"  {kpis['occupancy_rate'] * 100:>{col}.2f}"
        row += f"  {kpis['avg_selling_price']:>{col}.2f}"
        row += f"  {kpis['avg_reward']:>{col}.4f}"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_comparison(
    checkpoint_path: Optional[str] = None,
    n_episodes:      int           = 20,
    cfg:             Optional[DQNConfig] = None,
    output_dir:      str           = "outputs",
) -> Dict[str, Dict]:
    """Run baseline vs DQN comparison and save results to disk.

    Evaluates all three baseline strategies.  If *checkpoint_path* is
    provided, the trained DQN agent is also loaded and included.

    Args:
        checkpoint_path (str | None): Path to a ``.pt`` checkpoint file.
                                      When ``None``, only baselines are run.
        n_episodes      (int):        Episodes per strategy. Default is 20.
        cfg             (DQNConfig | None): Config for the DQN agent.
                                      Defaults to ``DQNConfig()`` if omitted.
        output_dir      (str):        Root output directory.
                                      CSV and JSON are saved here.
                                      Default is ``"outputs"``.

    Returns:
        dict: ``{strategy_name: kpi_dict}`` for every strategy evaluated.
    """
    if cfg is None:
        cfg = DQNConfig()

    os.makedirs(output_dir, exist_ok=True)

    # ── Build strategy list ──────────────────────────────────────────────
    strategies: List[BaseStrategy] = [
        FixedPriceStrategy(action=0, n_actions=cfg.action_size),  # lowest price
        FixedPriceStrategy(action=2, n_actions=cfg.action_size),  # mid price
        FixedPriceStrategy(action=4, n_actions=cfg.action_size),  # highest price
        RandomPriceStrategy(n_actions=cfg.action_size, seed=42),
        RuleBasedPricingStrategy(n_actions=cfg.action_size),
    ]

    # ── Optionally load DQN agent ────────────────────────────────────────
    if checkpoint_path is not None:
        # Deferred import to avoid circular dependency and allow running
        # comparison without torch when only baselines are needed.
        from src.agents.dqn_agent import DQNAgent
        from src.utils.checkpointing import load_checkpoint

        agent = DQNAgent(
            state_size=cfg.state_size,
            action_size=cfg.action_size,
            hidden_size=cfg.hidden_size,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
        )
        load_checkpoint(agent, checkpoint_path)
        strategies.append(_DQNStrategyAdapter(agent, n_actions=cfg.action_size))

    print("=" * 60)
    print("  Baseline vs DQN Strategy Comparison")
    print(f"  Strategies : {len(strategies)}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Max steps  : {cfg.max_steps_per_episode}")
    print("=" * 60)

    # ── Run all strategies ───────────────────────────────────────────────
    results = compare_strategies(
        strategies=strategies,
        n_episodes=n_episodes,
        max_steps=cfg.max_steps_per_episode,
    )

    # ── Print table ──────────────────────────────────────────────────────
    _print_comparison_table(results)

    # ── Save outputs ─────────────────────────────────────────────────────
    csv_path  = os.path.join(output_dir, "comparison_results.csv")
    json_path = os.path.join(output_dir, "comparison_results.json")

    _save_csv(results, csv_path)
    _save_json(results, json_path, n_episodes, cfg.max_steps_per_episode, checkpoint_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Usage: python -m src.evaluation.comparison [checkpoint_path] [n_episodes]
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    run_comparison(checkpoint_path=ckpt, n_episodes=n)
