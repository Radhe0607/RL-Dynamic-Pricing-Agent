"""
visualization.py
----------------
Reward and training-metric visualization for the DQN Dynamic Pricing Agent.

All plots are saved as PNG files inside ``outputs/plots/`` by default.
No plot is ever shown interactively (``plt.show()`` is intentionally
omitted) so this module is safe to use in headless / server environments.

Public API
----------
    plot_episode_rewards(rewards, save_dir, filename, window)
        Raw episode rewards + overlaid moving-average line.

    plot_moving_average(rewards, window, save_dir, filename)
        Moving-average reward only — cleaner view of the trend.

    plot_training_loss(losses, save_dir, filename, window)
        Per-episode TD loss + smoothed overlay.

    plot_epsilon_decay(epsilons, save_dir, filename)
        Epsilon curve across training.

    plot_training_dashboard(metrics_dict, save_dir)
        2×2 subplot dashboard combining all four signals in one figure.

    plot_eval_rewards(rewards, save_dir, filename)
        Bar chart of greedy evaluation rewards with mean/std bands.

Usage
-----
    from src.evaluation.visualization import plot_training_dashboard
    from src.evaluation.metrics import TrainingMetrics

    # After training
    plot_training_dashboard(metrics.as_dict(), save_dir="outputs/plots")
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Internal style constants
# ---------------------------------------------------------------------------

# Colour palette — chosen for readability on both light and dark backgrounds
_C_RAW      = "#90CAF9"   # light blue  — raw / noisy signal
_C_SMOOTH   = "#1565C0"   # deep blue   — smoothed / moving average
_C_LOSS     = "#EF9A9A"   # soft red    — loss (raw)
_C_LOSS_SM  = "#B71C1C"   # dark red    — loss (smoothed)
_C_EPSILON  = "#A5D6A7"   # soft green  — epsilon
_C_MEAN     = "#FF8F00"   # amber       — mean line in eval plot
_C_BG       = "#F8F9FA"   # near-white  — axes background

_DPI        = 150          # output resolution
_FIG_W      = 10           # default figure width  (inches)
_FIG_H      = 4            # default figure height (inches)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_moving_average(values: List[float], window: int) -> np.ndarray:
    """Compute a centred moving average using a uniform convolution kernel.

    For the first ``window-1`` elements where a full window is not yet
    available, the average is computed over the available prefix — this
    avoids introducing NaNs at the start of the curve.

    Args:
        values (list[float]): Input signal (one value per episode).
        window (int):         Smoothing window width.

    Returns:
        np.ndarray: Smoothed signal of the same length as *values*.
    """
    arr    = np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    # "same" keeps the output length equal to the input length;
    # we use np.convolve and then correct edge effects manually.
    smoothed = np.convolve(arr, kernel, mode="full")[: len(arr)]

    # Correct the left edge where the full window was not available
    for i in range(min(window - 1, len(arr))):
        smoothed[i] = arr[: i + 1].mean()

    return smoothed


def _ensure_dir(path: str) -> str:
    """Create *path* (and any parents) if it does not exist.

    Args:
        path (str): Directory path to create.

    Returns:
        str: The same *path*, for chaining convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path


def _apply_common_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent grid, labels, and background to a single Axes object.

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
    # Use integer episode numbers on the x-axis
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))


def _save_and_close(fig: plt.Figure, filepath: str) -> None:
    """Save the figure to *filepath* as PNG and close it to free memory.

    Args:
        fig      (plt.Figure): Figure to save.
        filepath (str):        Absolute or relative output path.
    """
    fig.savefig(filepath, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {filepath}")


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_episode_rewards(
    rewards: List[float],
    save_dir: str = "outputs/plots",
    filename: str = "episode_rewards.png",
    window: int = 50,
) -> str:
    """Plot raw episode rewards with a smoothed moving-average overlay.

    Two series are drawn on the same axes:
      - A thin, semi-transparent line showing the noisy per-episode reward.
      - A thicker, opaque line showing the rolling average (window episodes).

    Args:
        rewards  (list[float]): Per-episode total rewards from training.
        save_dir (str):         Directory where the PNG is written.
                                Created automatically if it does not exist.
        filename (str):         Output filename (must end in ``.png``).
        window   (int):         Moving-average smoothing window.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not rewards:
        print("  [plot] No rewards to plot — skipping episode_rewards.")
        return ""

    episodes = list(range(1, len(rewards) + 1))
    smoothed = _compute_moving_average(rewards, window)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_common_style(ax, "Episode Rewards", "Episode", "Total Reward")

    # Raw signal — faint background trace
    ax.plot(
        episodes, rewards,
        color=_C_RAW, linewidth=0.8, alpha=0.5,
        label="Raw reward",
    )
    # Smoothed signal — primary focus
    ax.plot(
        episodes, smoothed,
        color=_C_SMOOTH, linewidth=2.0,
        label=f"Moving avg (window={window})",
    )

    # Shade the area under the smoothed curve
    ax.fill_between(episodes, smoothed, alpha=0.12, color=_C_SMOOTH)

    # Mark the best moving-average point
    best_idx = int(np.argmax(smoothed))
    ax.scatter(
        episodes[best_idx], smoothed[best_idx],
        color=_C_SMOOTH, s=60, zorder=5,
        label=f"Best avg: {smoothed[best_idx]:.2f} (ep {episodes[best_idx]})",
    )

    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_and_close(fig, filepath)
    return filepath


def plot_moving_average(
    rewards: List[float],
    window: int = 50,
    save_dir: str = "outputs/plots",
    filename: str = "moving_average_reward.png",
) -> str:
    """Plot only the moving-average reward curve (no raw noise).

    This is a cleaner view than ``plot_episode_rewards`` and is easier to
    read when the raw signal is very noisy.

    Args:
        rewards  (list[float]): Per-episode total rewards from training.
        window   (int):         Smoothing window width.
        save_dir (str):         Output directory (created if missing).
        filename (str):         Output PNG filename.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not rewards:
        print("  [plot] No rewards to plot — skipping moving_average.")
        return ""

    episodes = list(range(1, len(rewards) + 1))
    smoothed = _compute_moving_average(rewards, window)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_common_style(
        ax,
        f"Moving Average Reward (window={window})",
        "Episode",
        f"Avg Reward (last {window} eps)",
    )

    ax.plot(episodes, smoothed, color=_C_SMOOTH, linewidth=2.2)
    ax.fill_between(episodes, smoothed, alpha=0.15, color=_C_SMOOTH)

    # Horizontal reference line at mean reward
    mean_val = float(np.mean(rewards))
    ax.axhline(
        mean_val, color="#FF8F00", linewidth=1.2,
        linestyle="--", label=f"Overall mean: {mean_val:.2f}",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_and_close(fig, filepath)
    return filepath


def plot_training_loss(
    losses: List[float],
    save_dir: str = "outputs/plots",
    filename: str = "training_loss.png",
    window: int = 50,
) -> str:
    """Plot per-episode mean TD loss with a smoothed overlay.

    Episodes where the replay buffer was still warming up will show 0.0
    loss — this is expected and rendered transparently in the plot.

    Args:
        losses   (list[float]): Per-episode mean TD loss values.
        save_dir (str):         Output directory (created if missing).
        filename (str):         Output PNG filename.
        window   (int):         Smoothing window for the overlay curve.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not losses:
        print("  [plot] No losses to plot — skipping training_loss.")
        return ""

    episodes = list(range(1, len(losses) + 1))
    smoothed = _compute_moving_average(losses, window)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_common_style(ax, "Training Loss (TD Error)", "Episode", "Mean Loss")

    ax.plot(episodes, losses,   color=_C_LOSS,    linewidth=0.8, alpha=0.45, label="Raw loss")
    ax.plot(episodes, smoothed, color=_C_LOSS_SM,  linewidth=2.0, label=f"Smoothed (window={window})")
    ax.fill_between(episodes, smoothed, alpha=0.1, color=_C_LOSS_SM)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_and_close(fig, filepath)
    return filepath


def plot_epsilon_decay(
    epsilons: List[float],
    save_dir: str = "outputs/plots",
    filename: str = "epsilon_decay.png",
) -> str:
    """Plot the epsilon (exploration rate) curve across training episodes.

    Args:
        epsilons (list[float]): Per-episode epsilon values after decay.
        save_dir (str):         Output directory (created if missing).
        filename (str):         Output PNG filename.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not epsilons:
        print("  [plot] No epsilon data to plot — skipping epsilon_decay.")
        return ""

    episodes = list(range(1, len(epsilons) + 1))

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_common_style(ax, "Epsilon (ε) Decay", "Episode", "ε (Exploration Rate)")

    ax.plot(episodes, epsilons, color=_C_EPSILON, linewidth=2.0)
    ax.fill_between(episodes, epsilons, alpha=0.2, color=_C_EPSILON)

    # Annotate final epsilon value
    final_eps = epsilons[-1]
    ax.axhline(
        final_eps, color="#388E3C", linewidth=1.0,
        linestyle="--", label=f"Final ε = {final_eps:.4f}",
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_and_close(fig, filepath)
    return filepath


def plot_training_dashboard(
    metrics_dict: dict,
    save_dir: str = "outputs/plots",
    filename: str = "training_dashboard.png",
    window: int = 50,
) -> str:
    """Render a 2×2 dashboard combining all four training signals in one figure.

    Panels (top-left → top-right → bottom-left → bottom-right):
      1. Episode Rewards    — raw + moving average
      2. Moving Average     — smoothed reward trend only
      3. Training Loss      — raw TD loss + smoothed overlay
      4. Epsilon Decay      — exploration rate curve

    Args:
        metrics_dict (dict): Dictionary returned by ``TrainingMetrics.as_dict()``.
                             Expected keys: ``"episode_rewards"``,
                             ``"episode_losses"``, ``"epsilons"``.
        save_dir     (str):  Output directory (created if missing).
        filename     (str):  Output PNG filename.
        window       (int):  Smoothing window used in all panels.

    Returns:
        str: Full path of the saved PNG file.
    """
    rewards  = metrics_dict.get("episode_rewards", [])
    losses   = metrics_dict.get("episode_losses",  [])
    epsilons = metrics_dict.get("epsilons",         [])

    if not rewards:
        print("  [plot] No data in metrics_dict — skipping dashboard.")
        return ""

    episodes = list(range(1, len(rewards) + 1))
    r_smooth = _compute_moving_average(rewards, window)
    l_smooth = _compute_moving_average(losses,  window) if losses else []

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("DQN Dynamic Pricing — Training Dashboard", fontsize=13, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#FFFFFF")

    # ── Panel 1: Episode Rewards ─────────────────────────────────────────
    ax = axes[0, 0]
    _apply_common_style(ax, "Episode Rewards", "Episode", "Total Reward")
    ax.plot(episodes, rewards,   color=_C_RAW,    linewidth=0.8, alpha=0.4, label="Raw")
    ax.plot(episodes, r_smooth,  color=_C_SMOOTH,  linewidth=2.0, label=f"Avg-{window}")
    ax.fill_between(episodes, r_smooth, alpha=0.12, color=_C_SMOOTH)
    ax.legend(fontsize=7)

    # ── Panel 2: Moving Average Only ─────────────────────────────────────
    ax = axes[0, 1]
    _apply_common_style(ax, f"Smoothed Reward (window={window})", "Episode", "Avg Reward")
    ax.plot(episodes, r_smooth, color=_C_SMOOTH, linewidth=2.2)
    ax.fill_between(episodes, r_smooth, alpha=0.15, color=_C_SMOOTH)
    mean_r = float(np.mean(rewards))
    ax.axhline(mean_r, color=_C_MEAN, linewidth=1.2, linestyle="--",
               label=f"Mean={mean_r:.2f}")
    ax.legend(fontsize=7)

    # ── Panel 3: Training Loss ────────────────────────────────────────────
    ax = axes[1, 0]
    _apply_common_style(ax, "Training Loss (TD Error)", "Episode", "Mean Loss")
    if losses:
        ax.plot(episodes, losses,   color=_C_LOSS,   linewidth=0.8, alpha=0.4, label="Raw")
        ax.plot(episodes, l_smooth, color=_C_LOSS_SM, linewidth=2.0, label=f"Avg-{window}")
        ax.fill_between(episodes, l_smooth, alpha=0.1, color=_C_LOSS_SM)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")

    # ── Panel 4: Epsilon Decay ────────────────────────────────────────────
    ax = axes[1, 1]
    _apply_common_style(ax, "Epsilon (ε) Decay", "Episode", "ε")
    if epsilons:
        ax.plot(episodes[:len(epsilons)], epsilons,
                color=_C_EPSILON, linewidth=2.0)
        ax.fill_between(episodes[:len(epsilons)], epsilons,
                        alpha=0.2, color=_C_EPSILON)
        ax.axhline(epsilons[-1], color="#388E3C", linewidth=1.0, linestyle="--",
                   label=f"Final ε={epsilons[-1]:.4f}")
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No epsilon data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")

    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_and_close(fig, filepath)
    return filepath


def plot_eval_rewards(
    rewards: List[float],
    save_dir: str = "outputs/plots",
    filename: str = "eval_rewards.png",
) -> str:
    """Bar chart of per-episode rewards from a greedy evaluation run.

    Draws individual episode bars with a horizontal mean line and a shaded
    ±1 std-dev band to make variance immediately visible.

    Args:
        rewards  (list[float]): Per-episode rewards from ``run_evaluation()``.
        save_dir (str):         Output directory (created if missing).
        filename (str):         Output PNG filename.

    Returns:
        str: Full path of the saved PNG file.
    """
    if not rewards:
        print("  [plot] No eval rewards to plot — skipping eval_rewards.")
        return ""

    episodes = list(range(1, len(rewards) + 1))
    arr      = np.array(rewards)
    mean_r   = float(arr.mean())
    std_r    = float(arr.std())

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _apply_common_style(ax, "Greedy Evaluation Rewards", "Episode", "Total Reward")

    # Individual bars
    bar_colors = [_C_SMOOTH if r >= mean_r else _C_RAW for r in rewards]
    ax.bar(episodes, rewards, color=bar_colors, alpha=0.75, width=0.6)

    # Mean line
    ax.axhline(mean_r, color=_C_MEAN, linewidth=2.0, linestyle="--",
               label=f"Mean = {mean_r:.3f}")

    # ±1 std band
    ax.axhspan(mean_r - std_r, mean_r + std_r,
               alpha=0.12, color=_C_MEAN, label=f"±1 std ({std_r:.3f})")

    ax.legend(fontsize=8)
    ax.set_xticks(episodes)
    fig.tight_layout()

    filepath = os.path.join(_ensure_dir(save_dir), filename)
    _save_and_close(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Convenience: save all training plots at once
# ---------------------------------------------------------------------------

def save_all_training_plots(
    metrics_dict: dict,
    save_dir: str = "outputs/plots",
    window: int = 50,
) -> List[str]:
    """Generate and save every training visualisation in one call.

    Produces five PNG files:
      - ``episode_rewards.png``
      - ``moving_average_reward.png``
      - ``training_loss.png``
      - ``epsilon_decay.png``
      - ``training_dashboard.png``

    Args:
        metrics_dict (dict): Output of ``TrainingMetrics.as_dict()``.
        save_dir     (str):  Output directory (created if missing).
        window       (int):  Moving-average window used in all plots.

    Returns:
        list[str]: Paths of all saved PNG files (empty strings skipped).
    """
    rewards  = metrics_dict.get("episode_rewards", [])
    losses   = metrics_dict.get("episode_losses",  [])
    epsilons = metrics_dict.get("epsilons",         [])

    paths = [
        plot_episode_rewards(rewards,  save_dir=save_dir, window=window),
        plot_moving_average (rewards,  save_dir=save_dir, window=window),
        plot_training_loss  (losses,   save_dir=save_dir, window=window),
        plot_epsilon_decay  (epsilons, save_dir=save_dir),
        plot_training_dashboard(metrics_dict, save_dir=save_dir, window=window),
    ]

    # Filter out empty strings returned when data was missing
    return [p for p in paths if p]
