"""
metrics.py
----------
Training evaluation metrics for the DQN Dynamic Pricing Agent.

Why a dedicated metrics module?
--------------------------------
The training loop (train.py) should focus on *control flow* — stepping
the environment, updating the agent, and saving checkpoints.  All
measurement and display concerns belong here, keeping each module
single-responsibility.

This module provides one public class:

    TrainingMetrics
        Stateful tracker that accumulates per-episode statistics and
        exposes helper methods consumed by the training loop.

Tracked signals
---------------
  - Episode reward          (raw + moving average over a configurable window)
  - Training loss           (per-episode mean + moving average)
  - Epsilon (ε)             (current exploration rate, full history)
  - Steps per episode       (how many env steps the episode lasted)
  - Buffer size             (snapshot at end of each episode)
  - Wall-clock elapsed time (since TrainingMetrics was constructed)

Usage
-----
    from src.evaluation.metrics import TrainingMetrics

    metrics = TrainingMetrics(window=50)

    for episode in range(1, max_episodes + 1):
        ...  # run episode
        metrics.record(
            episode=episode,
            reward=episode_reward,
            loss=avg_loss,
            epsilon=epsilon,
            steps=steps_in_episode,
            buffer_size=len(buffer),
        )
        if episode % log_every == 0:
            metrics.print_progress(episode, max_episodes)

    metrics.print_summary()
"""

import time
from collections import deque
from typing import Dict, List, Optional


class TrainingMetrics:
    """
    Stateful accumulator for DQN training statistics.

    All history lists grow by one element per ``record()`` call so that
    the full training curve is available for post-hoc analysis or plotting.
    Moving averages are computed over a sliding window (default 50 episodes)
    using an O(1) running-sum approach — no NumPy dependency required here.

    Args:
        window (int): Size of the rolling window used for moving averages.
                      Typical values: 50 (faster feedback) or 100 (smoother).
    """

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._start_time: float = time.time()

        # ── Full history (one value per episode) ───────────────────────────
        self.episode_rewards:  List[float] = []
        self.episode_losses:   List[float] = []
        self.epsilons:         List[float] = []
        self.steps_per_episode: List[int]  = []
        self.buffer_sizes:     List[int]   = []

        # ── Rolling-window accumulators (O(1) mean calculation) ────────────
        self._reward_window: deque = deque(maxlen=window)
        self._loss_window:   deque = deque(maxlen=window)

        # ── Running best reward (for tracking agent improvement) ───────────
        self._best_avg_reward: float = float("-inf")
        self._best_episode:    int   = 0

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------

    def record(
        self,
        episode: int,
        reward: float,
        loss: float,
        epsilon: float,
        steps: int,
        buffer_size: int,
    ) -> None:
        """Record all metrics for a completed episode.

        Call this once per episode, **after** the episode rollout finishes
        and before calling ``print_progress``.

        Args:
            episode     (int):   Completed episode number (1-indexed).
            reward      (float): Total undiscounted reward accumulated
                                 over the episode.
            loss        (float): Mean TD loss across all gradient steps
                                 taken during this episode (0.0 if the
                                 buffer was not yet warm).
            epsilon     (float): Current ε value **after** decay for
                                 this episode.
            steps       (int):   Number of environment steps taken in
                                 this episode.
            buffer_size (int):   Number of transitions currently stored
                                 in the replay buffer.
        """
        # Append to full history
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        self.epsilons.append(epsilon)
        self.steps_per_episode.append(steps)
        self.buffer_sizes.append(buffer_size)

        # Update rolling windows
        self._reward_window.append(reward)
        self._loss_window.append(loss)

        # Track best moving-average reward so far
        current_avg = self.moving_avg_reward()
        if current_avg > self._best_avg_reward:
            self._best_avg_reward = current_avg
            self._best_episode    = episode

    # ------------------------------------------------------------------
    # Moving average helpers
    # ------------------------------------------------------------------

    def moving_avg_reward(self) -> float:
        """Return the mean reward over the last ``window`` episodes.

        Returns:
            float: Moving average reward, or 0.0 if no episodes recorded yet.
        """
        if not self._reward_window:
            return 0.0
        return sum(self._reward_window) / len(self._reward_window)

    def moving_avg_loss(self) -> float:
        """Return the mean loss over the last ``window`` episodes.

        Returns:
            float: Moving average loss, or 0.0 if no episodes recorded yet.
        """
        if not self._loss_window:
            return 0.0
        return sum(self._loss_window) / len(self._loss_window)

    # ------------------------------------------------------------------
    # Epsilon helpers
    # ------------------------------------------------------------------

    def current_epsilon(self) -> float:
        """Return the most recently recorded epsilon value.

        Returns:
            float: Last epsilon, or 1.0 if no episodes recorded yet.
        """
        return self.epsilons[-1] if self.epsilons else 1.0

    def epsilon_progress(self) -> float:
        """Return how much ε has decayed as a percentage of total range.

        Progress = (ε_start − ε_current) / (ε_start − ε_end) × 100

        A value of 100 % means epsilon has fully annealed to ε_end.
        Because TrainingMetrics does not hold the config, this convenience
        is computed from the first and last recorded epsilon values.

        Returns:
            float: Decay progress in [0, 100], or 0.0 before any recording.
        """
        if len(self.epsilons) < 2:
            return 0.0
        epsilon_start = self.epsilons[0]
        epsilon_end   = self.epsilons[-1]
        total_range   = epsilon_start - epsilon_end
        if total_range <= 0:
            return 100.0
        decayed = epsilon_start - self.current_epsilon()
        return min(100.0, (decayed / total_range) * 100.0)

    # ------------------------------------------------------------------
    # Elapsed time
    # ------------------------------------------------------------------

    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds since this TrainingMetrics was created."""
        return time.time() - self._start_time

    def elapsed_str(self) -> str:
        """Return elapsed time as a human-readable HH:MM:SS string."""
        total = int(self.elapsed_seconds())
        h, remainder = divmod(total, 3600)
        m, s         = divmod(remainder, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # ------------------------------------------------------------------
    # Formatted console output
    # ------------------------------------------------------------------

    def print_progress(self, episode: int, max_episodes: int) -> None:
        """Print a single formatted progress line for the current episode.

        Format::

            [Ep  200/1000] reward=  12.50  avg50=  10.30  loss=0.0412
                           ε=0.367  buf=  2048  time=00:32

        Args:
            episode      (int): Current episode number.
            max_episodes (int): Total number of training episodes.
        """
        ep_reward   = self.episode_rewards[-1] if self.episode_rewards else 0.0
        avg_reward  = self.moving_avg_reward()
        avg_loss    = self.moving_avg_loss()
        epsilon     = self.current_epsilon()
        buf_size    = self.buffer_sizes[-1]   if self.buffer_sizes    else 0
        elapsed     = self.elapsed_str()

        # Highlight if this episode achieved a new best moving average
        marker = " ★" if (len(self.episode_rewards) > 1
                          and self._best_episode == episode) else ""

        print(
            f"  [Ep {episode:>5}/{max_episodes}]"
            f"  reward={ep_reward:>8.2f}"
            f"  avg{self._window}={avg_reward:>8.2f}"
            f"  loss={avg_loss:.4f}"
            f"  ε={epsilon:.3f}"
            f"  buf={buf_size:>6}"
            f"  time={elapsed}"
            f"{marker}"
        )

    def print_summary(self) -> None:
        """Print a full end-of-training statistics summary.

        Displays:
          - Total episodes trained
          - Total environment steps
          - Best / worst / final episode reward
          - Best moving-average reward and the episode it occurred at
          - Final epsilon and decay progress
          - Total wall-clock training time
        """
        if not self.episode_rewards:
            print("  [metrics] No episodes recorded — nothing to summarise.")
            return

        total_steps   = sum(self.steps_per_episode)
        best_reward   = max(self.episode_rewards)
        worst_reward  = min(self.episode_rewards)
        final_reward  = self.episode_rewards[-1]
        final_epsilon = self.current_epsilon()
        ep_decay      = self.epsilon_progress()

        separator = "=" * 62
        print(separator)
        print("  Training Summary")
        print(separator)
        print(f"  Episodes trained    : {len(self.episode_rewards)}")
        print(f"  Total env steps     : {total_steps:,}")
        print(f"  Best episode reward : {best_reward:.2f}")
        print(f"  Worst episode reward: {worst_reward:.2f}")
        print(f"  Final episode reward: {final_reward:.2f}")
        print(f"  Best avg-{self._window} reward : {self._best_avg_reward:.2f}"
              f"  (episode {self._best_episode})")
        print(f"  Final ε             : {final_epsilon:.4f}"
              f"  ({ep_decay:.1f} % decayed)")
        print(f"  Total training time : {self.elapsed_str()}")
        print(separator)

    # ------------------------------------------------------------------
    # Data export (for plotting / analysis)
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, list]:
        """Export all recorded history as a plain dictionary.

        Useful for saving metrics to JSON, plotting with matplotlib, or
        passing to a notebook for post-hoc analysis.

        Returns:
            dict: Keys are metric names; values are lists (one per episode).
        """
        return {
            "episode_rewards":   self.episode_rewards,
            "episode_losses":    self.episode_losses,
            "epsilons":          self.epsilons,
            "steps_per_episode": self.steps_per_episode,
            "buffer_sizes":      self.buffer_sizes,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of episodes recorded so far."""
        return len(self.episode_rewards)

    def __repr__(self) -> str:
        return (
            f"TrainingMetrics("
            f"episodes={len(self)}, "
            f"window={self._window}, "
            f"avg_reward={self.moving_avg_reward():.2f})"
        )
