"""
progress_monitor.py
--------------------
Training Progress Monitor for the DQN Dynamic Pricing Agent.

This module is the **single responsibility** display layer for the
training loop.  It consumes data from ``TrainingMetrics`` and renders
rich, human-readable progress panels to the terminal — without
owning any state or performing any computation of its own.

Separation of concerns
-----------------------
  metrics.py           — accumulates stats, computes moving averages (data layer)
  progress_monitor.py  — reads from metrics, renders output (display layer)
  train.py             — orchestrates training (control layer)

Features
--------
  Progress bar          — ASCII bar showing completion percentage.
  Episode panel         — current ep, total eps, reward, avg reward, epsilon.
  ETA                   — estimated time remaining based on elapsed rate.
  Milestone banners     — printed at 25 %, 50 %, 75 %, 100 % completion.
  Best-reward marker    — ★ marker when a new moving-average best is reached.
  Configurable interval — print every N episodes (default: 10).

Public API
----------
    ProgressMonitor(total_episodes, window, print_every, bar_width)
        Constructor.

    monitor.update(episode, reward, avg_reward, epsilon, loss,
                   steps, buffer_size)
        Call once per completed episode.  Prints output when the
        print interval fires.

    monitor.force_print(episode)
        Force a panel print regardless of the interval.

    monitor.print_header()
        Print the header banner at the start of training.

    monitor.print_footer(total_duration)
        Print the final summary banner at the end of training.

Usage (integration pattern)
----------------------------
    from src.training.progress_monitor import ProgressMonitor

    monitor = ProgressMonitor(total_episodes=cfg.max_episodes)
    monitor.print_header(cfg)

    for episode in range(1, cfg.max_episodes + 1):
        ...  # run episode, call metrics.record(...)
        monitor.update(
            episode=episode,
            reward=episode_reward,
            avg_reward=metrics.moving_avg_reward(),
            epsilon=epsilon,
            loss=avg_loss,
            steps=steps_taken,
            buffer_size=len(buffer),
        )

    monitor.print_footer(total_duration=time.time() - start_time)
"""

from __future__ import annotations

import time
from typing import Optional

from src.config.dqn_config import DQNConfig


# ---------------------------------------------------------------------------
# Internal style helpers
# ---------------------------------------------------------------------------

# Milestone checkpoints (as fractions of total_episodes) that trigger a
# special banner when crossed for the first time.
_MILESTONES = [0.25, 0.50, 0.75, 1.00]


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Render an ASCII progress bar string.

    Example output::

        [███████████████░░░░░░░░░░░░░░░]  48%

    Args:
        current (int): Number of completed steps.
        total   (int): Total number of steps.
        width   (int): Character width of the bar interior. Default 30.

    Returns:
        str: Formatted bar string including surrounding brackets and percentage.
    """
    pct   = current / max(total, 1)
    filled = int(pct * width)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct * 100:5.1f}%"


def _eta_str(elapsed: float, current: int, total: int) -> str:
    """Estimate remaining training time from elapsed rate.

    Args:
        elapsed (float): Wall-clock seconds elapsed so far.
        current (int):   Episodes completed so far.
        total   (int):   Total episodes to train.

    Returns:
        str: Human-readable ETA string, e.g. ``"ETA 04:32"`` or ``"ETA --:--"``.
    """
    if current <= 0:
        return "ETA --:--"
    rate = elapsed / current          # seconds per episode
    remaining_eps = total - current
    remaining_sec = int(rate * remaining_eps)
    m, s = divmod(remaining_sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"ETA {h:02d}:{m:02d}:{s:02d}"
    return f"ETA {m:02d}:{s:02d}"


def _elapsed_str(seconds: float) -> str:
    """Format *seconds* as HH:MM:SS or MM:SS.

    Args:
        seconds (float): Elapsed seconds.

    Returns:
        str: Formatted string.
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# ProgressMonitor
# ---------------------------------------------------------------------------

class ProgressMonitor:
    """Terminal progress display for the DQN training loop.

    Reads statistics from the training loop and renders formatted output
    at a configurable interval.  Holds no training state of its own —
    all data is supplied by the caller via ``update()``.

    Args:
        total_episodes (int):  Total number of training episodes.
        window         (int):  Moving-average window size (for display label).
                               Default 50.
        print_every    (int):  Print an update every N episodes.
                               Default 10.
        bar_width      (int):  Width of the ASCII progress bar interior.
                               Default 32.
    """

    def __init__(
        self,
        total_episodes: int,
        window:         int = 50,
        print_every:    int = 10,
        bar_width:      int = 32,
    ) -> None:
        self._total        = total_episodes
        self._window       = window
        self._print_every  = print_every
        self._bar_width    = bar_width

        # Internal state — wall-clock tracking only
        self._start_time: float = time.time()

        # Track which milestone banners have already been printed
        self._milestones_shown: set = set()

        # Running best moving-average reward (for ★ marker)
        self._best_avg_reward: float = float("-inf")

        # Episode count at which the current best was achieved
        self._best_episode: int = 0

    # ------------------------------------------------------------------
    # Header / footer
    # ------------------------------------------------------------------

    def print_header(self, cfg: Optional[DQNConfig] = None) -> None:
        """Print the training start banner.

        Args:
            cfg (DQNConfig | None): If provided, key hyperparameters are
                                    displayed in the header panel.
        """
        w   = 64
        sep = "═" * w

        print(f"╔{sep}╗")
        print(f"║  {'DQN Dynamic Pricing — Training Progress Monitor':<{w - 4}}║")
        print(f"╠{sep}╣")

        if cfg is not None:
            def _row(label: str, val) -> None:
                print(f"║  {label:<28}: {str(val):<{w - 36}}║")

            _row("Episodes",         cfg.max_episodes)
            _row("Steps / episode",  cfg.max_steps_per_episode)
            _row("Learning rate",    cfg.learning_rate)
            _row("Gamma (γ)",        cfg.gamma)
            _row("Tau (τ)",          cfg.tau)
            _row("Batch size",       cfg.batch_size)
            _row("Buffer capacity",  cfg.buffer_capacity)
            _row("Epsilon start",    cfg.epsilon_start)
            _row("Epsilon end",      cfg.epsilon_end)
            _row("Epsilon decay",    cfg.epsilon_decay)
            _row("Save every",       f"{cfg.save_every} episodes")
            print(f"╠{sep}╣")

        print(f"║  {'Training started …':<{w - 4}}║")
        print(f"╚{sep}╝")
        print()

    def print_footer(self, total_duration: float) -> None:
        """Print the end-of-training summary banner.

        Args:
            total_duration (float): Total wall-clock training duration in seconds.
        """
        w   = 64
        sep = "═" * w

        print()
        print(f"╔{sep}╗")
        print(f"║  {'Training Complete':<{w - 4}}║")
        print(f"╠{sep}╣")

        def _row(label: str, val: str) -> None:
            print(f"║  {label:<28}: {val:<{w - 36}}║")

        _row("Total episodes",     str(self._total))
        _row("Best avg reward",    f"{self._best_avg_reward:.4f}  (ep {self._best_episode})")
        _row("Total duration",     _elapsed_str(total_duration))

        print(f"╚{sep}╝")

    # ------------------------------------------------------------------
    # Per-episode update (main public method)
    # ------------------------------------------------------------------

    def update(
        self,
        episode:     int,
        reward:      float,
        avg_reward:  float,
        epsilon:     float,
        loss:        float    = 0.0,
        steps:       int      = 0,
        buffer_size: int      = 0,
    ) -> None:
        """Record one completed episode and print progress if the interval fires.

        Call this **once per episode**, after the rollout and after
        ``TrainingMetrics.record()`` has been called.

        Args:
            episode     (int):   Current episode number (1-indexed).
            reward      (float): Total undiscounted reward for this episode.
            avg_reward  (float): Moving-average reward over the last N episodes.
            epsilon     (float): Current epsilon value after decay.
            loss        (float): Mean TD loss for this episode. Default 0.0.
            steps       (int):   Environment steps taken in this episode.
            buffer_size (int):   Current replay buffer occupancy.
        """
        # Update running best
        if avg_reward > self._best_avg_reward:
            self._best_avg_reward = avg_reward
            self._best_episode    = episode

        # Check for milestone banners
        self._check_milestones(episode)

        # Print progress panel at the configured interval
        if episode % self._print_every == 0 or episode == 1:
            self._print_panel(
                episode=episode,
                reward=reward,
                avg_reward=avg_reward,
                epsilon=epsilon,
                loss=loss,
                steps=steps,
                buffer_size=buffer_size,
            )

    def force_print(
        self,
        episode:     int,
        reward:      float,
        avg_reward:  float,
        epsilon:     float,
        loss:        float = 0.0,
        steps:       int   = 0,
        buffer_size: int   = 0,
    ) -> None:
        """Force a progress panel print regardless of the print interval.

        Useful for printing the very last episode even if it is not a
        multiple of ``print_every``.

        Args:
            episode     (int):   Current episode number.
            reward      (float): Episode reward.
            avg_reward  (float): Moving-average reward.
            epsilon     (float): Current epsilon.
            loss        (float): Mean TD loss. Default 0.0.
            steps       (int):   Steps taken in the episode. Default 0.
            buffer_size (int):   Current buffer occupancy. Default 0.
        """
        self._print_panel(episode, reward, avg_reward, epsilon,
                          loss, steps, buffer_size)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _print_panel(
        self,
        episode:     int,
        reward:      float,
        avg_reward:  float,
        epsilon:     float,
        loss:        float,
        steps:       int,
        buffer_size: int,
    ) -> None:
        """Render a full progress panel to stdout.

        Panel layout::

            ──────────────────────────────────────────────────────────────
            Episode    200 / 1000     [████████░░░░░░░░░░░░░░░░░░░░░░]  20.0%
            ──────────────────────────────────────────────────────────────
              Episode Reward    :         0.00
              Avg Reward (50)   :         0.00   ★ (new best)
              Epsilon (ε)       :       0.3679
              Loss              :       0.0000
              Steps             :          200
              Buffer Size       :         2048
              Elapsed           :        00:12   ETA 00:48
            ──────────────────────────────────────────────────────────────

        Args:
            episode     (int):   Current episode.
            reward      (float): Episode reward.
            avg_reward  (float): Moving average reward.
            epsilon     (float): Current epsilon.
            loss        (float): Mean TD loss.
            steps       (int):   Steps in this episode.
            buffer_size (int):   Replay buffer size.
        """
        elapsed    = time.time() - self._start_time
        pct        = episode / max(self._total, 1) * 100
        bar        = _progress_bar(episode, self._total, self._bar_width)
        eta        = _eta_str(elapsed, episode, self._total)
        elapsed_s  = _elapsed_str(elapsed)

        is_best    = (episode == self._best_episode and episode > 1)
        best_marker = "  ★ new best" if is_best else ""

        sep  = "─" * 62
        w    = 18   # value column width

        print(sep)
        print(
            f"  Episode {episode:>6} / {self._total:<6}   "
            f"{bar}"
        )
        print(sep)

        def _row(label: str, val: str, note: str = "") -> None:
            print(f"  {label:<22}: {val:>{w}}{note}")

        _row("Episode Reward",         f"{reward:>12.4f}")
        _row(f"Avg Reward ({self._window})", f"{avg_reward:>12.4f}", best_marker)
        _row("Epsilon (ε)",            f"{epsilon:>12.6f}")
        _row("Completion",             f"{pct:>11.1f} %")
        _row("Loss",                   f"{loss:>12.6f}")
        _row("Steps",                  f"{steps:>12d}")
        _row("Buffer Size",            f"{buffer_size:>12d}")
        _row("Elapsed",                f"{elapsed_s:>12}",   f"   {eta}")

        print(sep)
        print()

    def _check_milestones(self, episode: int) -> None:
        """Print a milestone banner when training crosses a checkpoint fraction.

        Banners are printed at 25 %, 50 %, 75 %, and 100 % completion.
        Each banner is shown at most once per training run.

        Args:
            episode (int): Current episode number.
        """
        pct = episode / max(self._total, 1)

        for milestone in _MILESTONES:
            if pct >= milestone and milestone not in self._milestones_shown:
                self._milestones_shown.add(milestone)
                label = f"{int(milestone * 100)} % complete"
                if milestone == 1.00:
                    label = "Training complete  🎉"
                self._print_milestone_banner(label, episode)
                break   # one banner per episode maximum

    def _print_milestone_banner(self, label: str, episode: int) -> None:
        """Print a visually distinct milestone banner.

        Args:
            label   (str): Text to display in the banner.
            episode (int): Episode number at which the milestone was reached.
        """
        w   = 62
        sep = "■" * w
        print()
        print(sep)
        print(f"  ★  MILESTONE  |  {label}  |  Episode {episode}")
        print(sep)
        print()
