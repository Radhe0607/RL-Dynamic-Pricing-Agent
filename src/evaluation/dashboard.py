"""
dashboard.py
------------
Model Performance Comparison Dashboard for the DQN Dynamic Pricing Agent.

This module generates a side-by-side visual and tabular comparison of every
pricing strategy in the project — the trained DQN agent plus all three
hand-crafted baselines — across five key business KPIs.

Strategies compared
--------------------
  DQN Agent              — trained neural-network policy (loaded from checkpoint)
  FixedPrice (low)       — always picks the lowest price level (action 0)
  FixedPrice (mid)       — always picks the mid price level  (action 2)
  FixedPrice (high)      — always picks the highest price level (action 4)
  RandomPrice            — uniform random action each step
  RuleBasedPricing       — urgency-ratio heuristic (inventory ÷ remaining days)

KPIs visualised
----------------
  Total Revenue          — sum of (price × demand) across all episodes
  Average Reward         — mean episode reward
  Occupancy Rate (%)     — total_bookings / (episodes × max_steps) × 100
  Booking Rate (%)       — fraction of steps where demand > 0
  Average Selling Price  — mean price charged across all steps and episodes

Outputs (saved to outputs/dashboard/)
---------------------------------------
  bar_total_revenue.png          — grouped bar chart
  bar_avg_reward.png             — grouped bar chart
  bar_occupancy_rate.png         — grouped bar chart
  bar_booking_rate.png           — grouped bar chart
  bar_avg_selling_price.png      — grouped bar chart
  dashboard_overview.png         — 2×3 combined subplot dashboard
  comparison_table.png           — rendered KPI table as an image
  dashboard_results.csv          — one row per strategy, all KPIs
  dashboard_results.json         — machine-readable with run metadata

Public API
----------
    run_dashboard(checkpoint_path, n_episodes, cfg, output_dir) → dict
        Main entry point. Runs all strategies, builds all charts, exports files.

    build_dashboard(results, output_dir) → list[str]
        Lower-level: accepts a pre-computed results dict and generates charts.

    export_dashboard_results(results, output_dir) → tuple[str, str]
        Saves CSV and JSON only (no charts).

Usage
-----
    # Baselines only (no checkpoint)
    python -m src.evaluation.dashboard

    # With DQN agent
    python -m src.evaluation.dashboard checkpoints/dqn_final.pt 20

    # From Python
    from src.evaluation.dashboard import run_dashboard
    run_dashboard(checkpoint_path="checkpoints/dqn_final.pt", n_episodes=20)
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.baselines.strategies import (
    BaseStrategy,
    FixedPriceStrategy,
    RandomPriceStrategy,
    RuleBasedPricingStrategy,
)
from src.config.dqn_config import DQNConfig
from src.evaluation.comparison import (
    _DQNStrategyAdapter,
    compare_strategies,
)


# ---------------------------------------------------------------------------
# Visual style constants — consistent with visualization.py
# ---------------------------------------------------------------------------

# Colour palette — one distinct colour per strategy (up to 6 strategies)
_PALETTE = [
    "#1565C0",   # deep blue      — DQN Agent
    "#2E7D32",   # dark green     — FixedPrice (low)
    "#6A1B9A",   # deep purple    — FixedPrice (mid)
    "#E65100",   # deep orange    — FixedPrice (high)
    "#B71C1C",   # dark red       — RandomPrice
    "#F57F17",   # dark amber     — RuleBasedPricing
]

_C_BG      = "#F8F9FA"   # axes background
_C_GRID    = "#DEE2E6"   # gridline colour
_DPI       = 150          # output resolution
_BAR_W     = 0.62         # bar width for single-strategy plots
_ALPHA     = 0.88         # bar fill transparency

# KPI display configuration — maps dict key → (chart title, axis label, scale)
# scale=100 converts a 0–1 fraction to a percentage display
_KPI_CONFIG: List[Tuple[str, str, str, float]] = [
    ("total_revenue",    "Total Revenue",        "Revenue (£)",     1.0),
    ("avg_reward",       "Average Reward",        "Reward",          1.0),
    ("occupancy_rate",   "Occupancy Rate",        "Occupancy (%)",   100.0),
    ("booking_rate",     "Booking Rate",          "Booking (%)",     100.0),
    ("avg_selling_price","Average Selling Price", "Price (£)",       1.0),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    """Create *path* and any missing parent directories.

    Args:
        path (str): Target directory path.

    Returns:
        str: The same *path* (for chaining convenience).
    """
    os.makedirs(path, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, filepath: str) -> str:
    """Save *fig* as a PNG to *filepath* and release memory.

    Args:
        fig      (plt.Figure): Matplotlib figure to save.
        filepath (str):        Full output file path.

    Returns:
        str: The filepath passed in (for convenience chaining).
    """
    fig.savefig(filepath, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [dashboard] Saved → {filepath}")
    return filepath


def _strategy_colours(names: List[str]) -> List[str]:
    """Map a list of strategy names to their palette colours.

    Strategies are assigned colours in palette order.  If more strategies
    are provided than palette entries, colours cycle.

    Args:
        names (list[str]): Strategy display names.

    Returns:
        list[str]: Hex colour codes, one per name.
    """
    return [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]


def _apply_bar_style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply consistent visual style to a bar-chart axes.

    Args:
        ax     (plt.Axes): Target axes.
        title  (str):      Chart title.
        ylabel (str):      Y-axis label.
    """
    ax.set_facecolor(_C_BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color=_C_GRID, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.tick_params(axis="y", labelsize=8)
    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Individual bar-chart plots
# ---------------------------------------------------------------------------

def plot_kpi_bar(
    results:    Dict[str, Dict[str, float]],
    kpi_key:    str,
    title:      str,
    ylabel:     str,
    scale:      float  = 1.0,
    save_dir:   str    = "outputs/dashboard",
    filename:   Optional[str] = None,
) -> str:
    """Generate a horizontal grouped bar chart for one KPI across all strategies.

    Args:
        results  (dict):  ``{strategy_name: kpi_dict}`` from ``compare_strategies``.
        kpi_key  (str):   Key in each kpi_dict to plot (e.g. ``"total_revenue"``).
        title    (str):   Chart title shown above the plot.
        ylabel   (str):   Y-axis label.
        scale    (float): Multiply raw values before plotting (e.g. 100 for %).
        save_dir (str):   Output directory.
        filename (str):   Override auto-generated filename.

    Returns:
        str: Full path of the saved PNG file.
    """
    names  = list(results.keys())
    values = [results[n].get(kpi_key, 0.0) * scale for n in names]
    colours = _strategy_colours(names)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    _apply_bar_style(ax, title, ylabel)

    x      = np.arange(len(names))
    bars   = ax.bar(x, values, width=_BAR_W, color=colours, alpha=_ALPHA,
                    edgecolor="white", linewidth=0.8)

    # Value label on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,.1f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, ha="right")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)

    fig.tight_layout()

    fname    = filename or f"bar_{kpi_key}.png"
    filepath = os.path.join(_ensure_dir(save_dir), fname)
    return _save_fig(fig, filepath)


# ---------------------------------------------------------------------------
# Combined 2×3 dashboard overview
# ---------------------------------------------------------------------------

def plot_dashboard_overview(
    results:  Dict[str, Dict[str, float]],
    save_dir: str = "outputs/dashboard",
    filename: str = "dashboard_overview.png",
) -> str:
    """Render all five KPI charts in a single 2×3 subplot figure.

    The sixth panel (bottom-right) displays a compact legend mapping
    strategy names to colours.

    Args:
        results  (dict): ``{strategy_name: kpi_dict}`` from ``compare_strategies``.
        save_dir (str):  Output directory.
        filename (str):  Output PNG filename.

    Returns:
        str: Full path of the saved PNG file.
    """
    names   = list(results.keys())
    colours = _strategy_colours(names)
    x       = np.arange(len(names))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Strategy Performance Comparison Dashboard",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.patch.set_facecolor("#FFFFFF")

    for idx, (kpi_key, title, ylabel, scale) in enumerate(_KPI_CONFIG):
        row, col = divmod(idx, 3)
        ax       = axes[row, col]
        values   = [results[n].get(kpi_key, 0.0) * scale for n in names]

        _apply_bar_style(ax, title, ylabel)

        bars = ax.bar(x, values, width=_BAR_W, color=colours,
                      alpha=_ALPHA, edgecolor="white", linewidth=0.7)

        # Compact value labels — skip if zero to reduce clutter
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{val:,.0f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(["" for _ in names])  # labels in legend panel
        ax.set_ylim(0, max(values) * 1.18 if values else 1)

    # ── Legend panel (bottom-right, index 5) ────────────────────────────
    legend_ax = axes[1, 2]
    legend_ax.set_facecolor(_C_BG)
    legend_ax.axis("off")
    legend_ax.set_title("Strategies", fontsize=11, fontweight="bold", pad=8)

    patches = [
        mpatches.Patch(color=colours[i], label=names[i], alpha=_ALPHA)
        for i in range(len(names))
    ]
    legend_ax.legend(
        handles=patches,
        loc="center",
        fontsize=9,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.6,
    )

    fig.tight_layout()
    filepath = os.path.join(_ensure_dir(save_dir), filename)
    return _save_fig(fig, filepath)


# ---------------------------------------------------------------------------
# Comparison table image
# ---------------------------------------------------------------------------

def plot_comparison_table(
    results:  Dict[str, Dict[str, float]],
    save_dir: str = "outputs/dashboard",
    filename: str = "comparison_table.png",
) -> str:
    """Render the KPI comparison data as a formatted table image.

    The table has one row per strategy and one column per KPI.
    The best value in each column is highlighted in bold green.

    Args:
        results  (dict): ``{strategy_name: kpi_dict}`` from ``compare_strategies``.
        save_dir (str):  Output directory.
        filename (str):  Output PNG filename.

    Returns:
        str: Full path of the saved PNG file.
    """
    names   = list(results.keys())
    n_rows  = len(names)

    # Column definitions — (kpi_key, header, scale, fmt)
    col_defs = [
        ("total_revenue",    "Total\nRevenue (£)",    1.0,   "{:,.0f}"),
        ("avg_reward",       "Avg\nReward",           1.0,   "{:.2f}"),
        ("occupancy_rate",   "Occupancy\n(%)",        100.0, "{:.1f}"),
        ("booking_rate",     "Booking\nRate (%)",     100.0, "{:.1f}"),
        ("avg_selling_price","Avg Selling\nPrice (£)", 1.0,  "{:.2f}"),
        ("std_reward",       "Reward\nStd Dev",       1.0,   "{:.2f}"),
    ]
    n_cols = len(col_defs)

    # Build cell values and detect best-in-column indices
    col_headers = [cd[1] for cd in col_defs]
    cell_vals   = []   # list of rows; each row is a list of formatted strings
    raw_vals    = []   # list of rows; raw floats for comparison

    for name in names:
        kpis = results[name]
        row_raw  = [kpis.get(cd[0], 0.0) * cd[2] for cd in col_defs]
        row_fmt  = [cd[3].format(v) for cd, v in zip(col_defs, row_raw)]
        cell_vals.append(row_fmt)
        raw_vals.append(row_raw)

    # Per-column best index (higher is better for all 5 KPIs)
    raw_arr    = np.array(raw_vals, dtype=np.float64)
    best_rows  = raw_arr.argmax(axis=0)  # shape: (n_cols,)

    # ── Draw table ────────────────────────────────────────────────────────
    fig_h = max(2.5, 0.55 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Row colours — alternating light bands + header
    row_colours = []
    for r in range(n_rows):
        if r % 2 == 0:
            row_colours.append(["#EEF2FF"] * n_cols)
        else:
            row_colours.append(["#FFFFFF"] * n_cols)

    table = ax.table(
        cellText=cell_vals,
        rowLabels=names,
        colLabels=col_headers,
        cellColours=row_colours,
        rowColours=[_PALETTE[i % len(_PALETTE)] for i in range(n_rows)],
        colColours=["#1E3A5F"] * n_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.8)

    # Style header cells (col labels)
    for col_idx in range(n_cols):
        cell = table[0, col_idx]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_facecolor("#1E3A5F")

    # Style row-label cells
    for row_idx, name in enumerate(names):
        cell = table[row_idx + 1, -1]   # row label column index is -1
        cell.set_text_props(color="white", fontsize=8, fontweight="bold")

    # Highlight best-in-column cells with a green border
    for col_idx, best_row in enumerate(best_rows):
        cell = table[best_row + 1, col_idx]   # +1 because row 0 is header
        cell.set_text_props(fontweight="bold", color="#1B5E20")
        cell.set_edgecolor("#2E7D32")
        cell.set_linewidth(2.0)

    ax.set_title(
        "Strategy Performance Comparison — KPI Summary\n"
        "(bold green border = best in column)",
        fontsize=10, fontweight="bold", pad=12, loc="center",
    )

    fig.tight_layout()
    filepath = os.path.join(_ensure_dir(save_dir), filename)
    return _save_fig(fig, filepath)


# ---------------------------------------------------------------------------
# Console comparison table
# ---------------------------------------------------------------------------

def print_dashboard_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted side-by-side KPI comparison table to stdout.

    Args:
        results (dict): ``{strategy_name: kpi_dict}`` from ``compare_strategies``.
    """
    sep = "=" * 100
    col = 14

    print(sep)
    print("  Strategy Performance Comparison Dashboard")
    print(sep)

    # Header row
    header = f"  {'Strategy':<30}"
    for _, title, _, _ in _KPI_CONFIG:
        header += f"  {title[:col]:>{col}}"
    print(header)
    print("  " + "-" * 96)

    for name, kpis in results.items():
        row = f"  {name:<30}"
        for kpi_key, _, _, scale in _KPI_CONFIG:
            val = kpis.get(kpi_key, 0.0) * scale
            row += f"  {val:>{col}.2f}"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# CSV and JSON export
# ---------------------------------------------------------------------------

def export_dashboard_results(
    results:    Dict[str, Dict[str, float]],
    output_dir: str = "outputs/dashboard",
) -> Tuple[str, str]:
    """Save dashboard results to CSV and JSON.

    CSV layout: one header row + one row per strategy.
    JSON layout: metadata envelope with full results dict.

    Args:
        results    (dict): ``{strategy_name: kpi_dict}``.
        output_dir (str):  Output directory (created if missing).

    Returns:
        tuple[str, str]: ``(csv_path, json_path)`` absolute paths.
    """
    _ensure_dir(output_dir)

    csv_path  = os.path.join(output_dir, "dashboard_results.csv")
    json_path = os.path.join(output_dir, "dashboard_results.json")

    # ── CSV ───────────────────────────────────────────────────────────────
    kpi_fields = [k for k, *_ in _KPI_CONFIG] + ["std_reward", "total_bookings"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["strategy"] + kpi_fields)
        writer.writeheader()
        for strategy_name, kpis in results.items():
            row = {"strategy": strategy_name}
            row.update({k: kpis.get(k, "") for k in kpi_fields})
            writer.writerow(row)

    print(f"  [dashboard] CSV  saved → {csv_path}")

    # ── JSON ──────────────────────────────────────────────────────────────
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_strategies": len(results),
        "kpis":         [k for k, *_ in _KPI_CONFIG],
        "strategies":   results,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  [dashboard] JSON saved → {json_path}")

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Lower-level builder: charts from a pre-computed results dict
# ---------------------------------------------------------------------------

def build_dashboard(
    results:    Dict[str, Dict[str, float]],
    output_dir: str = "outputs/dashboard",
) -> List[str]:
    """Generate all chart PNGs from a pre-computed results dict.

    Use this when you already have ``results`` from ``compare_strategies``
    and only need the visual output.

    Args:
        results    (dict): ``{strategy_name: kpi_dict}``.
        output_dir (str):  Directory for all PNG outputs.

    Returns:
        list[str]: Paths of every saved PNG file.
    """
    _ensure_dir(output_dir)
    paths: List[str] = []

    # ── Five individual KPI bar charts ───────────────────────────────────
    for kpi_key, title, ylabel, scale in _KPI_CONFIG:
        p = plot_kpi_bar(
            results,
            kpi_key=kpi_key,
            title=title,
            ylabel=ylabel,
            scale=scale,
            save_dir=output_dir,
        )
        paths.append(p)

    # ── Combined 2×3 overview dashboard ──────────────────────────────────
    paths.append(plot_dashboard_overview(results, save_dir=output_dir))

    # ── Rendered comparison table ─────────────────────────────────────────
    paths.append(plot_comparison_table(results, save_dir=output_dir))

    return [p for p in paths if p]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dashboard(
    checkpoint_path: Optional[str] = None,
    n_episodes:      int            = 20,
    cfg:             Optional[DQNConfig] = None,
    output_dir:      str            = "outputs/dashboard",
) -> Dict[str, Dict[str, float]]:
    """Run all strategies, build all charts, print console table, export files.

    This is the single entry point for the comparison dashboard pipeline.

    Args:
        checkpoint_path (str | None): Path to a DQN ``.pt`` checkpoint.
                                      When ``None``, only baselines are run.
        n_episodes      (int):        Episodes per strategy. Default 20.
        cfg             (DQNConfig | None): Agent config. Uses ``DQNConfig()``
                                      defaults if omitted.
        output_dir      (str):        Root output directory.
                                      Default is ``"outputs/dashboard"``.

    Returns:
        dict: ``{strategy_name: kpi_dict}`` for every strategy evaluated.
    """
    if cfg is None:
        cfg = DQNConfig()

    # ── Build strategy list ───────────────────────────────────────────────
    strategies: List[BaseStrategy] = [
        FixedPriceStrategy(action=0, n_actions=cfg.action_size),   # low
        FixedPriceStrategy(action=2, n_actions=cfg.action_size),   # mid
        FixedPriceStrategy(action=4, n_actions=cfg.action_size),   # high
        RandomPriceStrategy(n_actions=cfg.action_size, seed=42),
        RuleBasedPricingStrategy(n_actions=cfg.action_size),
    ]

    # ── Optionally load the DQN agent ────────────────────────────────────
    if checkpoint_path is not None:
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
        # Prepend so DQN appears first in charts
        strategies.insert(0, _DQNStrategyAdapter(agent, n_actions=cfg.action_size))

    print("=" * 65)
    print("  Model Performance Comparison Dashboard")
    print(f"  Strategies : {len(strategies)}")
    print(f"  Episodes   : {n_episodes}  |  Max steps: {cfg.max_steps_per_episode}")
    print("=" * 65)

    # ── Run all strategies ────────────────────────────────────────────────
    results = compare_strategies(
        strategies=strategies,
        n_episodes=n_episodes,
        max_steps=cfg.max_steps_per_episode,
    )

    # ── Console table ─────────────────────────────────────────────────────
    print_dashboard_table(results)

    # ── Charts ────────────────────────────────────────────────────────────
    print(f"\n  [dashboard] Generating charts → {output_dir}/")
    build_dashboard(results, output_dir=output_dir)

    # ── CSV + JSON ───────────────────────────────────────────────────────
    export_dashboard_results(results, output_dir=output_dir)

    print("\n  [dashboard] Done.")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Usage: python -m src.evaluation.dashboard [checkpoint_path] [n_episodes]
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    run_dashboard(checkpoint_path=ckpt, n_episodes=n)
