"""
env_visualization.py
--------------------
Environment State Visualization for the DQN Dynamic Pricing Agent.

This module records how the environment evolves over the course of one
or more episodes and renders four time-series charts:

    1. Price vs Time        — what price was set at each step
    2. Demand vs Time       — how many units were demanded at that price
    3. Occupancy vs Time    — cumulative fraction of capacity sold
    4. Revenue vs Time      — cumulative revenue earned each step

All plots are saved as PNG files inside ``outputs/visualizations/``.
No interactive window is ever opened (``plt.show()`` is intentionally
omitted), making this module safe for headless / server environments.

Separation of concerns
-----------------------
  visualization.py      — training-curve plots (rewards, loss, epsilon)
  env_visualization.py  — environment-state time-series plots (this file)

Public API
----------
    collect_episode_data(strategy_or_agent, env, max_steps, n_episodes)
        Roll out episodes and collect per-step Price/Demand/Occupancy/Revenue.

    plot_price_vs_time(price_series, ...)
        Line chart of price at each time step.

    plot_demand_vs_time(demand_series, ...)
        Line chart of demand at each time step.

    plot_occupancy_vs_time(occupancy_series, ...)
        Cumulative occupancy (fraction of capacity filled) over time.

    plot_revenue_vs_time(revenue_series, ...)
        Cumulative revenue earned as the episode progresses.

    plot_env_dashboard(episode_data, ...)
        2×2 dashboard combining all four charts in one figure.

    visualize_environment(strategy_or_agent, env, ..., save_dir)
        One-call convenience: collects data + saves all five PNGs.

Integration with evaluate.py
------------------------------
    from src.evaluation.env_visualization import visualize_environment
    from src.evaluation.evaluate import run_evaluation

    # After evaluation, visualize the DQN agent's environment interaction
    visualize_environment(agent, env, save_dir="outputs/visualizations")

Usage (standalone)
------------------
    python -m src.evaluation.env_visualization checkpoints/dqn_final.pt
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.demand.demand_simulator import generate_demand
from src.environment.pricing_env import PricingEnvironment
from src.demand.config.config import MAX_PRICE, MIN_PRICE


# ---------------------------------------------------------------------------
# Colour palette — consistent with visualization.py
# ---------------------------------------------------------------------------

_C_PRICE     = "#1565C0"   # deep blue      — price line
_C_DEMAND    = "#6A1B9A"   # deep purple    — demand line
_C_OCCUPANCY = "#E65100"   # deep orange    — occupancy line
_C_REVENUE   = "#2E7D32"   # dark green     — revenue line
_C_FILL      = 0.13        # fill alpha shared across all charts
_C_BG        = "#F8F9FA"   # near-white     — axes background
_C_MEAN      = "#FF8F00"   # amber          — mean/reference lines

_DPI   = 150   # output resolution
_FIG_W = 10    # default single-plot width  (inches)
_FIG_H = 4     # default single-plot height (inches)

# Number of discrete price actions (matches PricingEnvironment.action_space)
_N_ACTIONS   = 5
_PRICE_STEP  = (MAX_PRICE - MIN_PRICE) / max(_N_ACTIONS - 1, 1)

# Assumed total capacity for occupancy calculation (units per episode)
# In the current stub the environment does not expose capacity, so we
# derive a reasonable proxy from the demand model's base demand.
_CAPACITY_PER_STEP = 100  # base_demand in demand_simulator.py


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _action_to_price(action: int, n_actions: int = _N_ACTIONS) -> float:
    """Convert a discrete action index to its price level.

    Args:
        action    (int): Action index in ``[0, n_actions)``.
        n_actions (int): Total number of discrete actions.

    Returns:
        float: Corresponding price value.
    """
    step = (MAX_PRICE - MIN_PRICE) / max(n_actions - 1, 1)
    return MIN_PRICE + action * step


def _ensure_dir(path: str) -> str:
    """Create *path* and any missing parent directories.

    Args:
        path (str): Target directory path.

    Returns:
        str: The same *path* (for chaining convenience).
    """
    os.makedirs(path, exist_ok=True)
    return path


def _apply_style(
    ax:     plt.Axes,
    title:  str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply a consistent visual style to an axes object.

    Args:
        ax     (plt.Axes): Target axes.
        title  (str):      Plot title.
        xlabel (str):      X-axis label.
        ylabel (str):      Y-axis label.
    """
    ax.set_facecolor(_C_BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))


def _save_fig(fig: plt.Figure, filepath: str) -> None:
    """Save *fig* as a PNG to *filepath* and release memory.

    Args:
        fig      (plt.Figure): Matplotlib figure to save.
        filepath (str):        Full output file path.
    """
    fig.savefig(filepath, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [env_viz] Saved → {filepath}")


def _mean_series(series_list: List[List[float]]) -> List[float]:
    """Compute the element-wise mean across multiple equal-length series.

    Args:
        series_list (list[list[float]]): Multiple per-episode time series.

    Returns:
        list[float]: Mean series (same length as each input series).
    """
    if not series_list:
        return []
    arr = np.array(series_list, dtype=np.float64)
    return arr.mean(axis=0).tolist()


# ---------------------------------------------------------------------------
# Episode data collection
# ---------------------------------------------------------------------------

def collect_episode_data(
    strategy_or_agent: Any,
    env:               PricingEnvironment,
    max_steps:         int = 200,
    n_episodes:        int = 1,
    n_actions:         int = _N_ACTIONS,
) -> Dict[str, List[List[float]]]:
    """Roll out *n_episodes* and record per-step environment metrics.

    At every time step the following signals are recorded:
      - **price**         : the price set by the strategy / agent
      - **demand**        : units demanded at that price (from demand model)
      - **occupancy**     : cumulative fraction of episode capacity sold
      - **revenue**       : cumulative revenue earned so far in the episode

    The strategy / agent is expected to expose a ``select_action(state)``
    or ``select_action(state, epsilon)`` method.  Both signatures are
    handled automatically.

    Args:
        strategy_or_agent : Any object with ``select_action(state[, epsilon])``
                            — a ``BaseStrategy`` or a ``DQNAgent``.
        env               (PricingEnvironment): Gymnasium environment instance.
        max_steps         (int): Hard cap on episode length. Default 200.
        n_episodes        (int): Number of episodes to collect. Default 1.
        n_actions         (int): Action-space size (for price conversion).

    Returns:
        dict: Keys are metric names; values are lists of per-episode series.

              ``{
                  "prices":     [[step0, step1, ...], ...],  # n_episodes lists
                  "demands":    [[step0, step1, ...], ...],
                  "occupancy":  [[step0, step1, ...], ...],  # cumulative fraction
                  "revenues":   [[step0, step1, ...], ...],  # cumulative
                  "steps":      [n_steps_ep0, n_steps_ep1, ...],
              }``
    """
    all_prices:    List[List[float]] = []
    all_demands:   List[List[float]] = []
    all_occupancy: List[List[float]] = []
    all_revenues:  List[List[float]] = []
    all_steps:     List[int]         = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        state    = np.array(state, dtype=np.float32)

        ep_prices:    List[float] = []
        ep_demands:   List[float] = []
        ep_occupancy: List[float] = []
        ep_revenues:  List[float] = []

        cumulative_bookings = 0
        cumulative_revenue  = 0.0
        total_capacity      = max_steps * _CAPACITY_PER_STEP  # episode ceiling

        for _ in range(max_steps):
            # ── Choose action ─────────────────────────────────────────────
            # Support both BaseStrategy.select_action(state) and
            # DQNAgent.select_action(state, epsilon)
            try:
                action = strategy_or_agent.select_action(state, 0.0)  # DQNAgent
            except TypeError:
                action = strategy_or_agent.select_action(state)        # BaseStrategy

            # ── Derive price and demand ───────────────────────────────────
            price  = _action_to_price(action, n_actions)
            demand = generate_demand(price)               # stochastic demand

            # Bookings = fulfilled demand (no inventory cap in the stub)
            bookings = max(0, demand)
            revenue  = bookings * price

            # ── Accumulate episode totals ─────────────────────────────────
            cumulative_bookings += bookings
            cumulative_revenue  += revenue
            occupancy_frac       = cumulative_bookings / max(total_capacity, 1)

            ep_prices.append(price)
            ep_demands.append(float(demand))
            ep_occupancy.append(min(1.0, occupancy_frac))  # cap at 100 %
            ep_revenues.append(cumulative_revenue)

            # ── Step the environment ──────────────────────────────────────
            next_state, _, terminated, truncated, _ = env.step(action)
            state = np.array(next_state, dtype=np.float32)

            if terminated or truncated:
                break

        all_prices.append(ep_prices)
        all_demands.append(ep_demands)
        all_occupancy.append(ep_occupancy)
        all_revenues.append(ep_revenues)
        all_steps.append(len(ep_prices))

    return {
        "prices":    all_prices,
        "demands":   all_demands,
        "occupancy": all_occupancy,
        "revenues":  all_revenues,
        "steps":     all_steps,
    }


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_price_vs_time(
    price_series: List[List[float]],
    save_dir:     str = "outputs/visualizations",
    filename:     str = "price_vs_time.png",
    label:        str = "Strategy",
) -> str:
    """Plot price at each time step over one or more episodes.

    When multiple episodes are provided, each episode is shown as a faint
    trace and the mean across episodes is drawn as the primary bold line.

    Args:
        price_series (list[list[float]]): Per-episode price sequences.
        save_dir     (str):              Output directory (created if missing).
        filename     (str):              Output PNG filename.
        label        (str):              Legend label for the strategy.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not price_series or not price_series[0]:
        print("  [env_viz] No price data — skipping price_vs_time.")
        return ""

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_style(ax, "Price vs Time", "Time Step", "Price (£)")

    # Draw each individual episode as a faint trace
    for ep_prices in price_series:
        steps = list(range(1, len(ep_prices) + 1))
        ax.plot(steps, ep_prices, color=_C_PRICE, linewidth=0.6, alpha=0.25)

    # Bold mean line
    mean_prices = _mean_series(price_series)
    steps       = list(range(1, len(mean_prices) + 1))
    ax.plot(steps, mean_prices, color=_C_PRICE, linewidth=2.2,
            label=f"{label} — mean price")
    ax.fill_between(steps, mean_prices, alpha=_C_FILL, color=_C_PRICE)

    # Reference lines at min and max price
    ax.axhline(MIN_PRICE, color="#B0BEC5", linewidth=1.0, linestyle=":",
               label=f"Min price ({MIN_PRICE})")
    ax.axhline(MAX_PRICE, color="#78909C", linewidth=1.0, linestyle=":",
               label=f"Max price ({MAX_PRICE})")

    ax.set_ylim(MIN_PRICE * 0.9, MAX_PRICE * 1.05)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_fig(fig, filepath)
    return filepath


def plot_demand_vs_time(
    demand_series: List[List[float]],
    save_dir:      str = "outputs/visualizations",
    filename:      str = "demand_vs_time.png",
    label:         str = "Strategy",
) -> str:
    """Plot demand at each time step over one or more episodes.

    Args:
        demand_series (list[list[float]]): Per-episode demand sequences.
        save_dir      (str):              Output directory (created if missing).
        filename      (str):              Output PNG filename.
        label         (str):              Legend label for the strategy.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not demand_series or not demand_series[0]:
        print("  [env_viz] No demand data — skipping demand_vs_time.")
        return ""

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_style(ax, "Demand vs Time", "Time Step", "Units Demanded")

    # Individual episode traces
    for ep_demands in demand_series:
        steps = list(range(1, len(ep_demands) + 1))
        ax.plot(steps, ep_demands, color=_C_DEMAND, linewidth=0.5, alpha=0.2)

    # Mean trace
    mean_demand = _mean_series(demand_series)
    steps       = list(range(1, len(mean_demand) + 1))
    ax.plot(steps, mean_demand, color=_C_DEMAND, linewidth=2.2,
            label=f"{label} — mean demand")
    ax.fill_between(steps, mean_demand, alpha=_C_FILL, color=_C_DEMAND)

    # Mean reference line
    overall_mean = float(np.mean([d for ep in demand_series for d in ep]))
    ax.axhline(overall_mean, color=_C_MEAN, linewidth=1.2, linestyle="--",
               label=f"Overall mean: {overall_mean:.1f}")

    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_fig(fig, filepath)
    return filepath


def plot_occupancy_vs_time(
    occupancy_series: List[List[float]],
    save_dir:         str = "outputs/visualizations",
    filename:         str = "occupancy_vs_time.png",
    label:            str = "Strategy",
) -> str:
    """Plot cumulative occupancy (fraction of capacity sold) over time.

    Occupancy is expressed as a percentage of the episode's total capacity.
    A perfect strategy would drive the curve to 100 % by the final step.

    Args:
        occupancy_series (list[list[float]]): Per-episode occupancy fractions
                                              (values in [0, 1]).
        save_dir         (str):               Output directory.
        filename         (str):               Output PNG filename.
        label            (str):               Legend label for the strategy.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not occupancy_series or not occupancy_series[0]:
        print("  [env_viz] No occupancy data — skipping occupancy_vs_time.")
        return ""

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_style(ax, "Cumulative Occupancy vs Time", "Time Step",
                 "Occupancy Rate (%)")

    # Individual episode traces (already fractional, convert to %)
    for ep_occ in occupancy_series:
        steps   = list(range(1, len(ep_occ) + 1))
        pct     = [v * 100 for v in ep_occ]
        ax.plot(steps, pct, color=_C_OCCUPANCY, linewidth=0.5, alpha=0.2)

    # Mean trace
    mean_occ = _mean_series(occupancy_series)
    steps    = list(range(1, len(mean_occ) + 1))
    mean_pct = [v * 100 for v in mean_occ]
    ax.plot(steps, mean_pct, color=_C_OCCUPANCY, linewidth=2.2,
            label=f"{label} — mean occupancy")
    ax.fill_between(steps, mean_pct, alpha=_C_FILL, color=_C_OCCUPANCY)

    # Reference at 100 % (full house)
    ax.axhline(100, color="#B0BEC5", linewidth=1.0, linestyle=":",
               label="Full capacity (100 %)")

    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_fig(fig, filepath)
    return filepath


def plot_revenue_vs_time(
    revenue_series: List[List[float]],
    save_dir:       str = "outputs/visualizations",
    filename:       str = "revenue_vs_time.png",
    label:          str = "Strategy",
) -> str:
    """Plot cumulative revenue earned as the episode progresses.

    The curve always starts at 0 and grows monotonically.  The slope
    at any point indicates the per-step revenue rate.

    Args:
        revenue_series (list[list[float]]): Per-episode cumulative revenue
                                            sequences.
        save_dir       (str):               Output directory.
        filename       (str):               Output PNG filename.
        label          (str):               Legend label for the strategy.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not revenue_series or not revenue_series[0]:
        print("  [env_viz] No revenue data — skipping revenue_vs_time.")
        return ""

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_style(ax, "Cumulative Revenue vs Time", "Time Step",
                 "Cumulative Revenue (£)")

    # Individual episode traces
    for ep_rev in revenue_series:
        steps = list(range(1, len(ep_rev) + 1))
        ax.plot(steps, ep_rev, color=_C_REVENUE, linewidth=0.5, alpha=0.2)

    # Mean trace
    mean_rev = _mean_series(revenue_series)
    steps    = list(range(1, len(mean_rev) + 1))
    ax.plot(steps, mean_rev, color=_C_REVENUE, linewidth=2.2,
            label=f"{label} — mean revenue")
    ax.fill_between(steps, mean_rev, alpha=_C_FILL, color=_C_REVENUE)

    # Annotate final mean revenue
    final_rev = mean_rev[-1] if mean_rev else 0
    ax.annotate(
        f"  Final: £{final_rev:,.0f}",
        xy=(steps[-1], final_rev),
        fontsize=8, color=_C_REVENUE, va="center",
    )

    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# 2×2 environment dashboard
# ---------------------------------------------------------------------------

def plot_env_dashboard(
    episode_data: Dict[str, List[List[float]]],
    save_dir:     str = "outputs/visualizations",
    filename:     str = "env_dashboard.png",
    label:        str = "Strategy",
) -> str:
    """Render a 2×2 dashboard combining all four environment-state plots.

    Panels:
      top-left  → Price vs Time
      top-right → Demand vs Time
      bottom-left  → Occupancy vs Time
      bottom-right → Revenue vs Time

    Args:
        episode_data (dict): Output of ``collect_episode_data()``.
        save_dir     (str):  Output directory (created if missing).
        filename     (str):  Output PNG filename.
        label        (str):  Strategy name shown in the figure title.

    Returns:
        str: Full path of the saved PNG file.
    """
    prices    = episode_data.get("prices",    [])
    demands   = episode_data.get("demands",   [])
    occupancy = episode_data.get("occupancy", [])
    revenues  = episode_data.get("revenues",  [])

    if not prices or not prices[0]:
        print("  [env_viz] No episode data — skipping env_dashboard.")
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Environment State — {label}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.patch.set_facecolor("#FFFFFF")

    n_eps = len(prices)

    # Helper — draw faint individual traces + bold mean on a given axes
    def _draw(ax, series, color, ylabel):
        if not series:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            return
        for ep in series:
            steps = list(range(1, len(ep) + 1))
            ax.plot(steps, ep, color=color, linewidth=0.5, alpha=0.2)
        mean = _mean_series(series)
        steps = list(range(1, len(mean) + 1))
        ax.plot(steps, mean, color=color, linewidth=2.2, label=f"Mean ({n_eps} eps)")
        ax.fill_between(steps, mean, alpha=_C_FILL, color=color)
        ax.legend(fontsize=7)

    # ── Top-left: Price ──────────────────────────────────────────────────
    ax = axes[0, 0]
    _apply_style(ax, "Price vs Time", "Step", "Price (£)")
    _draw(ax, prices, _C_PRICE, "Price (£)")
    ax.axhline(MIN_PRICE, color="#B0BEC5", linewidth=0.8, linestyle=":")
    ax.axhline(MAX_PRICE, color="#78909C", linewidth=0.8, linestyle=":")
    ax.set_ylim(MIN_PRICE * 0.9, MAX_PRICE * 1.05)

    # ── Top-right: Demand ────────────────────────────────────────────────
    ax = axes[0, 1]
    _apply_style(ax, "Demand vs Time", "Step", "Units Demanded")
    _draw(ax, demands, _C_DEMAND, "Demand")
    ax.set_ylim(bottom=0)

    # ── Bottom-left: Occupancy ───────────────────────────────────────────
    ax = axes[1, 0]
    _apply_style(ax, "Cumulative Occupancy vs Time", "Step", "Occupancy (%)")
    occ_pct = [[v * 100 for v in ep] for ep in occupancy]
    _draw(ax, occ_pct, _C_OCCUPANCY, "Occupancy (%)")
    ax.axhline(100, color="#B0BEC5", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 110)

    # ── Bottom-right: Revenue ────────────────────────────────────────────
    ax = axes[1, 1]
    _apply_style(ax, "Cumulative Revenue vs Time", "Step", "Revenue (£)")
    _draw(ax, revenues, _C_REVENUE, "Revenue (£)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# One-call convenience entry point
# ---------------------------------------------------------------------------

def visualize_environment(
    strategy_or_agent: Any,
    env:               Optional[PricingEnvironment] = None,
    n_episodes:        int  = 5,
    max_steps:         int  = 200,
    save_dir:          str  = "outputs/visualizations",
    label:             str  = "Agent",
    n_actions:         int  = _N_ACTIONS,
) -> List[str]:
    """Collect episode data and save all five environment-state PNGs.

    This is the single integration point for the evaluation pipeline.
    Pass any strategy or DQN agent and this function handles the rest.

    Saved files::

        outputs/visualizations/
            price_vs_time.png
            demand_vs_time.png
            occupancy_vs_time.png
            revenue_vs_time.png
            env_dashboard.png

    Args:
        strategy_or_agent : A ``BaseStrategy`` or ``DQNAgent`` — any object
                            with a ``select_action(state[, epsilon])`` method.
        env               (PricingEnvironment | None): Environment to evaluate
                            in. A fresh ``PricingEnvironment()`` is created
                            automatically when ``None``.
        n_episodes        (int): Episodes to collect for averaging.
                                 Default is 5.
        max_steps         (int): Hard cap on episode length. Default is 200.
        save_dir          (str): Output directory. Default ``outputs/visualizations``.
        label             (str): Human-readable strategy name used in titles.
        n_actions         (int): Action-space size. Default is 5.

    Returns:
        list[str]: Paths of all saved PNG files (empty strings omitted).
    """
    if env is None:
        env = PricingEnvironment()

    print(f"  [env_viz] Collecting {n_episodes} episode(s) for '{label}' …")
    data = collect_episode_data(
        strategy_or_agent=strategy_or_agent,
        env=env,
        max_steps=max_steps,
        n_episodes=n_episodes,
        n_actions=n_actions,
    )

    print(f"  [env_viz] Saving plots to {save_dir}/ …")
    paths = [
        plot_price_vs_time    (data["prices"],    save_dir=save_dir, label=label),
        plot_demand_vs_time   (data["demands"],   save_dir=save_dir, label=label),
        plot_occupancy_vs_time(data["occupancy"], save_dir=save_dir, label=label),
        plot_revenue_vs_time  (data["revenues"],  save_dir=save_dir, label=label),
        plot_env_dashboard    (data, save_dir=save_dir, label=label),
    ]

    return [p for p in paths if p]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Usage: python -m src.evaluation.env_visualization [checkpoint_path]
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None

    if checkpoint:
        # Visualize the trained DQN agent
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
        visualize_environment(agent, n_episodes=10, label="DQN Agent")
    else:
        # Visualize the Rule-Based baseline (no checkpoint required)
        from src.baselines.strategies import RuleBasedPricingStrategy
        strategy = RuleBasedPricingStrategy()
        visualize_environment(strategy, n_episodes=10, label="Rule-Based")
