"""
report.py
---------
Training report generator for the DQN Dynamic Pricing Agent.

After a training run completes, this module computes a comprehensive
set of statistics from the recorded metrics and writes them to a
human-readable ``.txt`` file in ``outputs/reports/``.

Why a dedicated report module?
-------------------------------
  - ``metrics.py``        — accumulates data *during* training (in-memory)
  - ``visualization.py``  — renders data as PNG charts
  - ``report.py``         — serialises data as a persistent text report
                            (this file)

Each module has a single, clear responsibility. The report is the
permanent, shareable artefact of a training run; the metrics object
exists only for the lifetime of the Python process.

Report contents
---------------
  Header     — project name, timestamp, report file path
  Run info   — checkpoint dir, config snapshot
  Training   — total episodes, total env steps, training duration
  Rewards    — best / worst / final / mean / std / median episode reward
  Learning   — mean & final training loss
  Exploration— initial ε, final ε, % decayed
  Top-5      — five highest-reward episodes

Public API
----------
    generate_report(metrics, cfg, report_dir, filename) → str
        Main entry point.  Pass a ``TrainingMetrics`` instance and a
        ``DQNConfig``; the report is written to disk and its path returned.

    build_report_text(metrics, cfg) → str
        Returns the full report as a plain string (no disk I/O).
        Useful for logging or embedding in a notebook.

Usage
-----
    from src.evaluation.report import generate_report
    from src.evaluation.metrics import TrainingMetrics
    from src.config.dqn_config import DQNConfig

    # After training
    generate_report(metrics, cfg)
"""

from __future__ import annotations

import os
import statistics
from datetime import datetime, timezone
from typing import Optional

# TrainingMetrics and DQNConfig are imported lazily inside functions
# (using TYPE_CHECKING guard) to keep the module importable even if the
# rest of the project is not installed / on the Python path.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.dqn_config import DQNConfig
    from src.evaluation.metrics import TrainingMetrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEPARATOR   = "=" * 64
_SECTION_SEP = "-" * 64
_PROJECT     = "RL Dynamic Pricing — DQN Agent"


def _fmt_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable HH:MM:SS string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted string, e.g. ``"01:23:45"`` or ``"05:12"``.
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _top_n_episodes(rewards: list, n: int = 5) -> list[tuple[int, float]]:
    """Return the top-*n* (episode_number, reward) pairs by reward value.

    Args:
        rewards (list[float]): Per-episode rewards (1-indexed episodes).
        n       (int):         Number of top episodes to return.

    Returns:
        list[tuple[int, float]]: Sorted descending by reward.
    """
    indexed = [(i + 1, r) for i, r in enumerate(rewards)]
    return sorted(indexed, key=lambda t: t[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Report text builder
# ---------------------------------------------------------------------------

def build_report_text(
    metrics: "TrainingMetrics",
    cfg:     "DQNConfig",
) -> str:
    """Compose the full training report as a plain-text string.

    This function is pure — it performs no file I/O.  All data is read
    from the ``metrics`` object and the ``cfg`` dataclass.

    Args:
        metrics (TrainingMetrics): Populated metrics object from a completed
                                   training run.
        cfg     (DQNConfig):       Hyperparameter config used during training.

    Returns:
        str: Multi-line report text ready to be written to a file or logged.
    """
    rewards = metrics.episode_rewards
    losses  = metrics.episode_losses
    eps     = metrics.epsilons
    steps   = metrics.steps_per_episode
    n_eps   = len(rewards)

    # ── Guard: nothing recorded ───────────────────────────────────────────
    if n_eps == 0:
        return "No training data recorded — report cannot be generated.\n"

    # ── Pre-compute statistics ────────────────────────────────────────────
    total_steps   = sum(steps)
    duration_secs = metrics.elapsed_seconds()
    duration_str  = _fmt_duration(duration_secs)

    reward_mean   = statistics.mean(rewards)
    reward_std    = statistics.stdev(rewards) if n_eps > 1 else 0.0
    reward_median = statistics.median(rewards)
    reward_best   = max(rewards)
    reward_worst  = min(rewards)
    reward_final  = rewards[-1]

    # Mean loss — exclude warm-up zeros so the number is meaningful
    non_zero_losses = [l for l in losses if l > 0.0]
    loss_mean  = statistics.mean(non_zero_losses)  if non_zero_losses else 0.0
    loss_final = losses[-1] if losses else 0.0

    epsilon_start  = eps[0]  if eps else cfg.epsilon_start
    epsilon_final  = eps[-1] if eps else cfg.epsilon_end
    epsilon_decayed = (
        ((epsilon_start - epsilon_final) / max(epsilon_start - cfg.epsilon_end, 1e-9)) * 100
        if (epsilon_start - cfg.epsilon_end) > 0
        else 100.0
    )

    top5 = _top_n_episodes(rewards, n=5)

    # ── Report timestamp ──────────────────────────────────────────────────
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Assemble lines ────────────────────────────────────────────────────
    lines: list[str] = [
        _SEPARATOR,
        f"  {_PROJECT}",
        f"  Training Report",
        _SEPARATOR,
        f"  Generated : {now}",
        f"  Checkpoint: {cfg.checkpoint_dir}/",
        "",

        # ── Run Configuration ─────────────────────────────────────────────
        _SECTION_SEP,
        "  RUN CONFIGURATION",
        _SECTION_SEP,
        f"  State size              : {cfg.state_size}",
        f"  Action size             : {cfg.action_size}",
        f"  Hidden layer size       : {cfg.hidden_size}",
        f"  Learning rate           : {cfg.learning_rate}",
        f"  Discount factor (γ)     : {cfg.gamma}",
        f"  Soft-update rate (τ)    : {cfg.tau}",
        f"  Replay buffer capacity  : {cfg.buffer_capacity:,}",
        f"  Mini-batch size         : {cfg.batch_size}",
        f"  Warm-up transitions     : {cfg.min_buffer_size:,}",
        f"  Max episodes            : {cfg.max_episodes:,}",
        f"  Max steps / episode     : {cfg.max_steps_per_episode:,}",
        f"  Checkpoint saved every  : {cfg.save_every} episodes",
        "",

        # ── Training Duration ─────────────────────────────────────────────
        _SECTION_SEP,
        "  TRAINING DURATION",
        _SECTION_SEP,
        f"  Total episodes trained  : {n_eps:,}",
        f"  Total environment steps : {total_steps:,}",
        f"  Wall-clock duration     : {duration_str}",
        f"  Avg steps / episode     : {total_steps / n_eps:.1f}",
        "",

        # ── Reward Statistics ─────────────────────────────────────────────
        _SECTION_SEP,
        "  REWARD STATISTICS",
        _SECTION_SEP,
        f"  Best episode reward     : {reward_best:>10.4f}",
        f"  Worst episode reward    : {reward_worst:>10.4f}",
        f"  Final episode reward    : {reward_final:>10.4f}",
        f"  Mean reward             : {reward_mean:>10.4f}",
        f"  Std deviation           : {reward_std:>10.4f}",
        f"  Median reward           : {reward_median:>10.4f}",
        f"  Best moving-avg reward  : {metrics._best_avg_reward:>10.4f}"
        f"  (episode {metrics._best_episode})",
        "",

        # ── Loss Statistics ───────────────────────────────────────────────
        _SECTION_SEP,
        "  LOSS STATISTICS  (TD / Bellman error)",
        _SECTION_SEP,
        f"  Mean training loss      : {loss_mean:>10.6f}",
        f"  Final episode loss      : {loss_final:>10.6f}",
        f"  Episodes with zero loss : {losses.count(0.0):,}"
        f"  (replay buffer warm-up)",
        "",

        # ── Exploration (ε) ───────────────────────────────────────────────
        _SECTION_SEP,
        "  EXPLORATION (ε-greedy)",
        _SECTION_SEP,
        f"  Initial ε               : {epsilon_start:.4f}",
        f"  Final ε                 : {epsilon_final:.4f}",
        f"  ε decayed               : {epsilon_decayed:.1f} %",
        f"  ε minimum (cfg)         : {cfg.epsilon_end:.4f}",
        f"  ε decay rate (cfg)      : {cfg.epsilon_decay}",
        "",

        # ── Top-5 Episodes ────────────────────────────────────────────────
        _SECTION_SEP,
        "  TOP 5 EPISODES BY REWARD",
        _SECTION_SEP,
        f"  {'Rank':<6}  {'Episode':>8}  {'Reward':>12}",
        f"  {'----':<6}  {'-------':>8}  {'------':>12}",
    ]

    for rank, (ep_num, ep_reward) in enumerate(top5, start=1):
        lines.append(f"  {rank:<6}  {ep_num:>8}  {ep_reward:>12.4f}")

    lines += [
        "",
        _SEPARATOR,
        "  End of Report",
        _SEPARATOR,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    metrics:    "TrainingMetrics",
    cfg:        "DQNConfig",
    report_dir: str = "outputs/reports",
    filename:   Optional[str] = None,
) -> str:
    """Generate and save a DQN training report to disk.

    Computes comprehensive statistics from the completed training run and
    writes a human-readable ``.txt`` file to *report_dir*.

    Args:
        metrics    (TrainingMetrics): Fully populated metrics object.
        cfg        (DQNConfig):       Config used during this training run.
        report_dir (str):             Directory where the report is saved.
                                      Created automatically if it does not exist.
                                      Default: ``"outputs/reports"``.
        filename   (str | None):      Override the auto-generated filename.
                                      Auto format:
                                      ``dqn_report_YYYYMMDD_HHMMSS.txt``.

    Returns:
        str: Absolute path of the saved report file.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    # ── Ensure output directory exists ────────────────────────────────────
    os.makedirs(report_dir, exist_ok=True)

    # ── Build filename ────────────────────────────────────────────────────
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"dqn_report_{timestamp}.txt"

    filepath = os.path.join(report_dir, filename)

    # ── Compose report text ───────────────────────────────────────────────
    report_text = build_report_text(metrics, cfg)

    # ── Write to disk ─────────────────────────────────────────────────────
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    print(f"  [report] Training report saved → {filepath}")
    return filepath
