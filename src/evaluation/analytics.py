"""
analytics.py
------------
Environment Performance Analytics Dashboard for the DQN Dynamic Pricing Agent.

This module takes the raw per-step episode data collected during evaluation
and computes a comprehensive set of business and RL performance metrics.
Results are displayed as a formatted console dashboard and exported to both
CSV and JSON in ``outputs/analytics/``.

Separation of concerns
-----------------------
  metrics.py       — accumulates RL stats *during* training (loss, epsilon …)
  env_viz.py       — collects raw per-step data and renders time-series plots
  report.py        — saves a post-training text summary (hyperparams, rewards)
  analytics.py     — computes aggregated KPIs from episode data (this file)

Metrics computed
-----------------
  avg_reward          : Mean total reward across all evaluation episodes.
  total_revenue       : Sum of all price × demand revenue across all episodes.
  avg_revenue         : Mean per-episode revenue.
  avg_occupancy_pct   : Mean final occupancy rate as a percentage (0–100).
  booking_rate        : Mean fraction of demand steps where demand > 0.
  avg_selling_price   : Mean price charged across all steps and episodes.
  demand_mean         : Mean demand per step across all episodes.
  demand_std          : Std deviation of per-step demand.
  demand_min          : Minimum per-step demand observed.
  demand_max          : Maximum per-step demand observed.
  n_episodes          : Number of episodes evaluated.
  total_steps         : Total number of environment steps taken.

Public API
----------
    compute_analytics(episode_data, rewards) → dict
        Pure computation — no I/O.  Returns all metrics as a dict.

    print_analytics_dashboard(analytics_dict, label)
        Prints a formatted, aligned console dashboard.

    export_analytics(analytics_dict, output_dir, label) → tuple[str, str]
        Saves CSV and JSON to *output_dir*.  Returns (csv_path, json_path).

    run_analytics(episode_data, rewards, label, output_dir) → dict
        One-call convenience: compute → print → export.

Usage
-----
    from src.evaluation.analytics import run_analytics
    from src.evaluation.env_visualization import collect_episode_data
    from src.environment.pricing_env import PricingEnvironment

    env  = PricingEnvironment()
    data = collect_episode_data(agent, env, max_steps=200, n_episodes=10)
    run_analytics(data, rewards=episode_rewards, label="DQN Agent")
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten(series_list: List[List[float]]) -> List[float]:
    """Flatten a list of per-episode series into a single list.

    Args:
        series_list (list[list[float]]): Per-episode data (e.g. per-step prices).

    Returns:
        list[float]: All values from all episodes in order.
    """
    return [v for episode in series_list for v in episode]


def _ensure_dir(path: str) -> str:
    """Create *path* and any missing parent directories.

    Args:
        path (str): Target directory.

    Returns:
        str: The same *path* (for chaining convenience).
    """
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_analytics(
    episode_data: Dict[str, List[List[float]]],
    rewards:      Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute environment performance analytics from collected episode data.

    This function is **pure** — it performs no I/O and has no side effects.
    All values are derived from the ``episode_data`` dict returned by
    ``collect_episode_data()`` in ``env_visualization.py``.

    Analytics computed
    ------------------
    avg_reward          Mean total reward across episodes (from *rewards* list;
                        0.0 if not supplied).
    total_revenue       Sum of final cumulative revenue across all episodes.
    avg_revenue         Mean per-episode final revenue.
    std_revenue         Std deviation of per-episode final revenue.
    avg_occupancy_pct   Mean final occupancy across episodes, as a %.
    booking_rate        Fraction of steps where demand > 0 (across all steps).
    avg_selling_price   Mean price charged across all steps and episodes.
    std_selling_price   Std deviation of prices charged.
    demand_mean         Mean demand per step across all episodes.
    demand_std          Std deviation of per-step demand.
    demand_min          Minimum per-step demand observed.
    demand_max          Maximum per-step demand observed.
    n_episodes          Number of episodes evaluated.
    total_steps         Total environment steps across all episodes.

    Args:
        episode_data (dict): Output of ``collect_episode_data()``.
                             Expected keys: ``"prices"``, ``"demands"``,
                             ``"occupancy"``, ``"revenues"``, ``"steps"``.
        rewards      (list[float] | None): Per-episode total rewards.
                             When ``None``, ``avg_reward`` is set to 0.0.

    Returns:
        dict[str, float]: Named KPI values.  All float, ready for JSON/CSV.
    """
    prices    = episode_data.get("prices",    [])
    demands   = episode_data.get("demands",   [])
    occupancy = episode_data.get("occupancy", [])
    revenues  = episode_data.get("revenues",  [])
    steps_per = episode_data.get("steps",     [])

    n_episodes   = len(prices)
    total_steps  = int(sum(steps_per)) if steps_per else 0

    # ── Rewards ──────────────────────────────────────────────────────────
    if rewards and len(rewards) > 0:
        avg_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards)) if len(rewards) > 1 else 0.0
    else:
        avg_reward = 0.0
        std_reward = 0.0

    # ── Revenue (take the last cumulative value in each episode) ─────────
    ep_revenues = [ep[-1] for ep in revenues if ep]   # final revenue per ep
    total_revenue = sum(ep_revenues)
    avg_revenue   = float(np.mean(ep_revenues))  if ep_revenues else 0.0
    std_revenue   = float(np.std(ep_revenues))   if len(ep_revenues) > 1 else 0.0

    # ── Occupancy (last cumulative fraction, convert to %) ───────────────
    ep_final_occ  = [ep[-1] * 100 for ep in occupancy if ep]
    avg_occ_pct   = float(np.mean(ep_final_occ)) if ep_final_occ else 0.0

    # ── Price statistics ─────────────────────────────────────────────────
    all_prices        = _flatten(prices)
    avg_selling_price = float(np.mean(all_prices))  if all_prices else 0.0
    std_selling_price = float(np.std(all_prices))   if all_prices else 0.0

    # ── Demand statistics ─────────────────────────────────────────────────
    all_demands  = _flatten(demands)
    demand_mean  = float(np.mean(all_demands))  if all_demands else 0.0
    demand_std   = float(np.std(all_demands))   if all_demands else 0.0
    demand_min   = float(min(all_demands))      if all_demands else 0.0
    demand_max   = float(max(all_demands))      if all_demands else 0.0

    # ── Booking rate = fraction of steps with demand > 0 ─────────────────
    positive_demand_steps = sum(1 for d in all_demands if d > 0)
    booking_rate = positive_demand_steps / max(len(all_demands), 1)

    return {
        # Core performance
        "avg_reward":          round(avg_reward,         4),
        "std_reward":          round(std_reward,         4),
        # Revenue
        "total_revenue":       round(total_revenue,      4),
        "avg_revenue":         round(avg_revenue,        4),
        "std_revenue":         round(std_revenue,        4),
        # Occupancy & bookings
        "avg_occupancy_pct":   round(avg_occ_pct,        4),
        "booking_rate":        round(booking_rate,       4),
        # Pricing
        "avg_selling_price":   round(avg_selling_price,  4),
        "std_selling_price":   round(std_selling_price,  4),
        # Demand
        "demand_mean":         round(demand_mean,        4),
        "demand_std":          round(demand_std,         4),
        "demand_min":          round(demand_min,         4),
        "demand_max":          round(demand_max,         4),
        # Run info
        "n_episodes":          n_episodes,
        "total_steps":         total_steps,
    }


# ---------------------------------------------------------------------------
# Console dashboard
# ---------------------------------------------------------------------------

def print_analytics_dashboard(
    analytics: Dict[str, float],
    label:     str = "Agent",
) -> None:
    """Print a formatted, aligned analytics dashboard to stdout.

    The dashboard groups metrics into four labelled sections:
      1. RL Performance     — reward statistics
      2. Revenue            — total and per-episode revenue
      3. Occupancy          — occupancy rate and booking rate
      4. Pricing & Demand   — price distribution and demand statistics

    Args:
        analytics (dict): Output of ``compute_analytics()``.
        label     (str):  Human-readable name for the agent / strategy.
    """
    sep     = "=" * 62
    sec_sep = "-" * 62

    print(sep)
    print(f"  Environment Performance Analytics  |  {label}")
    print(f"  Episodes: {analytics['n_episodes']}  |  "
          f"Total steps: {analytics['total_steps']:,}")
    print(sep)

    # ── RL Performance ────────────────────────────────────────────────────
    print(f"  {'RL PERFORMANCE':}")
    print(sec_sep)
    print(f"  {'Average reward':<35}: {analytics['avg_reward']:>10.4f}")
    print(f"  {'Reward std deviation':<35}: {analytics['std_reward']:>10.4f}")
    print()

    # ── Revenue ───────────────────────────────────────────────────────────
    print(f"  {'REVENUE':}")
    print(sec_sep)
    print(f"  {'Total revenue (all episodes)':<35}: "
          f"{analytics['total_revenue']:>10,.2f}")
    print(f"  {'Average revenue per episode':<35}: "
          f"{analytics['avg_revenue']:>10,.2f}")
    print(f"  {'Revenue std deviation':<35}: "
          f"{analytics['std_revenue']:>10,.2f}")
    print()

    # ── Occupancy & Bookings ──────────────────────────────────────────────
    print(f"  {'OCCUPANCY & BOOKINGS':}")
    print(sec_sep)
    print(f"  {'Average occupancy rate':<35}: "
          f"{analytics['avg_occupancy_pct']:>9.2f} %")
    print(f"  {'Booking rate (demand > 0 steps)':<35}: "
          f"{analytics['booking_rate'] * 100:>9.2f} %")
    print()

    # ── Pricing & Demand ──────────────────────────────────────────────────
    print(f"  {'PRICING & DEMAND':}")
    print(sec_sep)
    print(f"  {'Average selling price':<35}: {analytics['avg_selling_price']:>10.4f}")
    print(f"  {'Price std deviation':<35}: {analytics['std_selling_price']:>10.4f}")
    print(f"  {'Mean demand per step':<35}: {analytics['demand_mean']:>10.4f}")
    print(f"  {'Demand std deviation':<35}: {analytics['demand_std']:>10.4f}")
    print(f"  {'Minimum demand':<35}: {analytics['demand_min']:>10.4f}")
    print(f"  {'Maximum demand':<35}: {analytics['demand_max']:>10.4f}")

    print(sep)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_analytics(
    analytics:  Dict[str, float],
    output_dir: str = "outputs/analytics",
    label:      str = "agent",
) -> Tuple[str, str]:
    """Save analytics results to CSV and JSON files.

    Both files are written to *output_dir* with the timestamp embedded in
    the filename so successive runs do not overwrite each other.

    File naming::

        dqn_agent_analytics_20250625_143022.csv
        dqn_agent_analytics_20250625_143022.json

    Args:
        analytics  (dict): Output of ``compute_analytics()``.
        output_dir (str):  Destination directory.
                           Created automatically if it does not exist.
                           Default is ``"outputs/analytics"``.
        label      (str):  Short name embedded in filenames.
                           Spaces replaced with underscores.

    Returns:
        tuple[str, str]: ``(csv_path, json_path)`` — absolute paths of the
                         saved files.
    """
    _ensure_dir(output_dir)

    # Sanitise label for use in filenames
    safe_label = label.lower().replace(" ", "_").replace("/", "-")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = f"{safe_label}_analytics_{timestamp}"

    csv_path  = os.path.join(output_dir, base_name + ".csv")
    json_path = os.path.join(output_dir, base_name + ".json")

    # ── CSV ───────────────────────────────────────────────────────────────
    # Each KPI is written as one row: metric_name, value
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])            # header
        for key, value in analytics.items():
            writer.writerow([key, value])

    print(f"  [analytics] CSV  saved → {csv_path}")

    # ── JSON ──────────────────────────────────────────────────────────────
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "label":        label,
        "analytics":    analytics,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  [analytics] JSON saved → {json_path}")

    return csv_path, json_path


# ---------------------------------------------------------------------------
# One-call convenience entry point
# ---------------------------------------------------------------------------

def run_analytics(
    episode_data: Dict[str, List[List[float]]],
    rewards:      Optional[List[float]] = None,
    label:        str                   = "Agent",
    output_dir:   str                   = "outputs/analytics",
) -> Dict[str, float]:
    """Compute, display, and export environment performance analytics.

    This is the single integration point for the evaluation pipeline.
    Internally it chains ``compute_analytics`` → ``print_analytics_dashboard``
    → ``export_analytics`` in one call.

    Args:
        episode_data (dict): Raw per-step data from ``collect_episode_data()``.
                             Keys: ``"prices"``, ``"demands"``, ``"occupancy"``,
                             ``"revenues"``, ``"steps"``.
        rewards      (list[float] | None): Per-episode total rewards from the
                             evaluation loop.  Pass ``None`` if not available.
        label        (str):  Human-readable name displayed in the dashboard
                             and embedded in output filenames.
                             Default is ``"Agent"``.
        output_dir   (str):  Directory for CSV and JSON exports.
                             Default is ``"outputs/analytics"``.

    Returns:
        dict[str, float]: The full analytics dictionary (same as
                          ``compute_analytics()``).
    """
    # Step 1 — Compute all KPIs
    analytics = compute_analytics(episode_data, rewards=rewards)

    # Step 2 — Print formatted dashboard to console
    print_analytics_dashboard(analytics, label=label)

    # Step 3 — Export to disk
    export_analytics(analytics, output_dir=output_dir, label=label)

    return analytics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Usage: python -m src.evaluation.analytics [checkpoint_path] [n_episodes]
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    n_eps      = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    from src.environment.pricing_env import PricingEnvironment
    from src.evaluation.env_visualization import collect_episode_data

    env = PricingEnvironment()

    if checkpoint:
        from src.agents.dqn_agent import DQNAgent
        from src.config.dqn_config import DQNConfig
        from src.utils.checkpointing import load_checkpoint

        cfg   = DQNConfig()
        agent = DQNAgent(
            state_size=cfg.state_size,
            action_size=cfg.action_size,
            hidden_size=cfg.hidden_size,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
        )
        load_checkpoint(agent, checkpoint)
        data = collect_episode_data(agent, env, max_steps=cfg.max_steps_per_episode,
                                    n_episodes=n_eps)
        run_analytics(data, label="DQN Agent")
    else:
        # Run with Rule-Based baseline (no checkpoint needed)
        from src.baselines.strategies import RuleBasedPricingStrategy
        strategy = RuleBasedPricingStrategy()
        data     = collect_episode_data(strategy, env, max_steps=200,
                                        n_episodes=n_eps)
        run_analytics(data, label="Rule-Based Baseline")
