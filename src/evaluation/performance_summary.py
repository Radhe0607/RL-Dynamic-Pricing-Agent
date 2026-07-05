"""
performance_summary.py
-----------------------
Training Performance Summary for the DQN Dynamic Pricing Agent.

This module generates a focused, at-a-glance performance summary
immediately after training completes.  It is intentionally distinct
from ``report.py``:

  report.py               — full audit document (config, loss, top-5 eps, …)
  performance_summary.py  — compact, human-readable summary panel (this file)

The summary is designed to answer one question at a glance:
    *"How well did the agent train?"*

Summary contents
----------------
  Total training episodes         — how long training ran
  Total environment steps         — total agent–environment interactions
  Best episode reward             — single-episode peak performance
  Average reward                  — mean over all episodes
  Final episode reward            — performance at the end of training
  Best moving-average reward      — smoothed peak (window configurable)
  Final epsilon (ε)               — exploration level at the end
  Epsilon decay progress          — how much ε annealed as a percentage
  Total training duration         — wall-clock time

Outputs
-------
  Console panel   — rich, aligned text printed to stdout immediately.
  Text file       — same content saved to ``outputs/reports/`` for later
                    reference, with a timestamp in the filename.

Public API
----------
    generate_performance_summary(metrics, cfg, total_duration,
                                 output_dir, filename) → str
        Main entry point.  Prints to console and saves to disk.

    build_summary_text(metrics, cfg, total_duration) → str
        Pure function — returns the summary as a string.  No I/O.

    print_summary_panel(metrics, cfg, total_duration) → None
        Prints the formatted console panel only (no file I/O).

Usage
-----
    from src.evaluation.performance_summary import generate_performance_summary

    # After training
    generate_performance_summary(
        metrics=metrics,
        cfg=cfg,
        total_duration=time.time() - start_time,
    )
"""

from __future__ import annotations

import os
import statistics
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.dqn_config import DQNConfig
    from src.evaluation.metrics import TrainingMetrics


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Convert elapsed seconds to a human-readable HH:MM:SS string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted string, e.g. ``"01h 23m 45s"`` or ``"05m 12s"``.
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _pct_bar(value: float, width: int = 20) -> str:
    """Render a small inline ASCII bar proportional to *value* (0–100 %).

    Args:
        value (float): Percentage value in [0, 100].
        width (int):   Total bar width in characters. Default 20.

    Returns:
        str: e.g. ``"[████████░░░░░░░░░░░░]  42.0%"``
    """
    pct    = max(0.0, min(100.0, value))
    filled = int(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


# ---------------------------------------------------------------------------
# Core summary builder (pure, no I/O)
# ---------------------------------------------------------------------------

def build_summary_text(
    metrics:        "TrainingMetrics",
    cfg:            "DQNConfig",
    total_duration: float = 0.0,
) -> str:
    """Compose the performance summary as a plain-text string.

    This function is **pure** — it performs no I/O and has no side effects.
    All data is derived from the ``metrics`` object and the ``cfg`` dataclass.

    Args:
        metrics        (TrainingMetrics): Completed training metrics object.
        cfg            (DQNConfig):       Config used during training.
        total_duration (float):           Total wall-clock training time in
                                          seconds. Defaults to the elapsed
                                          time reported by *metrics* when 0.

    Returns:
        str: Multi-line summary text, suitable for printing or writing to disk.
    """
    rewards = metrics.episode_rewards
    eps_hist = metrics.epsilons
    steps    = metrics.steps_per_episode
    n_eps    = len(rewards)

    # ── Guard: no episodes recorded ───────────────────────────────────────
    if n_eps == 0:
        return (
            "=" * 64 + "\n"
            "  Training Performance Summary\n"
            "=" * 64 + "\n"
            "  No episodes were recorded — summary cannot be generated.\n"
            "=" * 64 + "\n"
        )

    # ── Aggregate statistics ───────────────────────────────────────────────
    total_steps   = sum(steps)
    duration_secs = total_duration if total_duration > 0.0 else metrics.elapsed_seconds()
    duration_str  = _fmt_duration(duration_secs)

    best_reward   = max(rewards)
    avg_reward    = statistics.mean(rewards)
    final_reward  = rewards[-1]
    best_avg      = getattr(metrics, "_best_avg_reward", avg_reward)
    best_avg_ep   = getattr(metrics, "_best_episode",   n_eps)

    epsilon_final = eps_hist[-1] if eps_hist else cfg.epsilon_end
    epsilon_start = eps_hist[0]  if eps_hist else cfg.epsilon_start
    total_range   = max(epsilon_start - cfg.epsilon_end, 1e-9)
    decayed_frac  = (epsilon_start - epsilon_final) / total_range
    epsilon_pct   = min(100.0, decayed_frac * 100.0)

    completion_pct = n_eps / max(cfg.max_episodes, 1) * 100.0

    # ── Timestamp ──────────────────────────────────────────────────────────
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Assemble lines ─────────────────────────────────────────────────────
    sep     = "=" * 64
    sub_sep = "-" * 64
    lbl_w   = 32   # label column width
    val_w   = 14   # value column width

    def _row(label: str, value: str, note: str = "") -> str:
        return f"  {label:<{lbl_w}}: {value:>{val_w}}{note}"

    lines = [
        sep,
        "  RL Dynamic Pricing — DQN Agent",
        "  Training Performance Summary",
        sep,
        f"  Generated : {now}",
        "",

        # ── Training Run ──────────────────────────────────────────────────
        sub_sep,
        "  TRAINING RUN",
        sub_sep,
        _row("Total episodes trained",    f"{n_eps:,}",
             f"  / {cfg.max_episodes:,}  {_pct_bar(completion_pct, width=16)}"),
        _row("Total environment steps",   f"{total_steps:,}"),
        _row("Avg steps per episode",     f"{total_steps / n_eps:.1f}"),
        _row("Total training duration",   duration_str),
        "",

        # ── Reward Performance ─────────────────────────────────────────────
        sub_sep,
        "  REWARD PERFORMANCE",
        sub_sep,
        _row("Best episode reward",       f"{best_reward:>14.4f}"),
        _row("Average reward",            f"{avg_reward:>14.4f}"),
        _row("Final episode reward",      f"{final_reward:>14.4f}"),
        _row(f"Best moving-avg reward",   f"{best_avg:>14.4f}",
             f"  (episode {best_avg_ep})"),
        "",

        # ── Exploration ────────────────────────────────────────────────────
        sub_sep,
        "  EXPLORATION (ε-greedy)",
        sub_sep,
        _row("Initial epsilon (ε)",       f"{epsilon_start:.6f}"),
        _row("Final epsilon (ε)",         f"{epsilon_final:.6f}"),
        _row("Epsilon decay progress",    f"{epsilon_pct:>13.1f}%",
             f"  {_pct_bar(epsilon_pct, width=16)}"),
        _row("Epsilon floor (cfg)",       f"{cfg.epsilon_end:.6f}"),
        "",

        sep,
        "  End of Performance Summary",
        sep,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Console printer (no file I/O)
# ---------------------------------------------------------------------------

def print_summary_panel(
    metrics:        "TrainingMetrics",
    cfg:            "DQNConfig",
    total_duration: float = 0.0,
) -> None:
    """Print the performance summary panel to stdout.

    This is a convenience wrapper around ``build_summary_text`` that
    prints directly to the terminal without writing any file.

    Args:
        metrics        (TrainingMetrics): Completed training metrics.
        cfg            (DQNConfig):       Config used during training.
        total_duration (float):           Total wall-clock time in seconds.
    """
    print(build_summary_text(metrics, cfg, total_duration))


# ---------------------------------------------------------------------------
# Public entry point (console + file)
# ---------------------------------------------------------------------------

def generate_performance_summary(
    metrics:        "TrainingMetrics",
    cfg:            "DQNConfig",
    total_duration: float       = 0.0,
    output_dir:     str         = "outputs/reports",
    filename:       Optional[str] = None,
) -> str:
    """Generate, print, and save the training performance summary.

    This is the single integration point for the post-training pipeline.
    It chains ``build_summary_text`` → print to console → write to disk.

    Args:
        metrics        (TrainingMetrics): Completed training metrics.
        cfg            (DQNConfig):       Config used during training.
        total_duration (float):           Total wall-clock training time in
                                          seconds.  Pass ``0.0`` to fall back
                                          to ``metrics.elapsed_seconds()``.
        output_dir     (str):             Directory for the text file.
                                          Created automatically if missing.
                                          Default: ``"outputs/reports"``.
        filename       (str | None):      Override the auto-generated filename.
                                          Auto format:
                                          ``performance_summary_YYYYMMDD_HHMMSS.txt``.

    Returns:
        str: Absolute path of the saved summary file.

    Raises:
        OSError: If the output directory cannot be created or the file
                 cannot be written.
    """
    # ── Build summary text ────────────────────────────────────────────────
    summary_text = build_summary_text(metrics, cfg, total_duration)

    # ── Print to console ──────────────────────────────────────────────────
    print(summary_text)

    # ── Save to disk ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"performance_summary_{timestamp}.txt"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(summary_text)

    print(f"  [summary] Performance summary saved → {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# CLI entry point — run standalone for any metrics JSON
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Demo mode: generate a synthetic summary with placeholder values
    print("  [performance_summary] Demo mode — generating synthetic summary …\n")

    # Lazy imports to allow standalone execution
    from src.config.dqn_config import DQNConfig
    from src.evaluation.metrics import TrainingMetrics

    cfg     = DQNConfig(max_episodes=500)
    metrics = TrainingMetrics(window=50)

    # Populate with synthetic data
    import random
    for ep in range(1, 501):
        metrics.record(
            episode=ep,
            reward=random.gauss(100.0, 30.0),
            loss=random.uniform(0.001, 0.05),
            epsilon=max(cfg.epsilon_end, cfg.epsilon_start * (cfg.epsilon_decay ** ep)),
            steps=200,
            buffer_size=min(ep * 200, cfg.buffer_capacity),
        )

    generate_performance_summary(metrics, cfg, total_duration=312.4)
