"""
logger.py
---------
Training Logger for the DQN Dynamic Pricing Agent.

This module provides a ``TrainingLogger`` class that wraps Python's
built-in ``logging`` module and emits structured training records to two
destinations simultaneously:

  1. **File handler** — appends to ``outputs/logs/training.log`` so that
     every training run is fully preserved on disk, even when the terminal
     is closed or output is piped.

  2. **Console handler** — prints INFO-level messages to stdout so the user
     sees progress in real time.

Log record format
------------------
File (includes timestamp and severity level)::

    2025-06-27 16:12:03,841 | INFO     | [Ep 0010/1000] reward=12345.00 loss=0.0421 eps=0.9048 steps=100 buf=640 dur=0.23s

Console (compact, no timestamp)::

    [Ep 0010/1000] reward=12345.00 loss=0.0421 eps=0.9048 steps=100 buf=640 dur=0.23s

Logger isolation
-----------------
``TrainingLogger`` always creates a *named* logger (``"dqn_training"``)
rather than using the root logger.  This prevents its handlers from being
inherited by or polluting other loggers in the project (e.g. matplotlib,
gymnasium internals).

Idempotent handler attachment
-------------------------------
If ``TrainingLogger`` is instantiated more than once (e.g. during testing or
when train.py is imported), the class checks whether the named logger already
has handlers and skips re-adding them.  This avoids duplicate log lines in
the console.

Public API
----------
    TrainingLogger(log_dir, log_filename, console_level, file_level)
        Constructor — creates output directory, attaches handlers.

    logger.log_episode(episode, max_episodes, reward, loss, epsilon,
                       steps, buffer_size, duration)
        Emit one structured line per training episode.

    logger.log_summary(total_episodes, mean_reward, best_reward,
                       final_epsilon, total_duration)
        Emit a multi-line summary block at the end of training.

    logger.log_checkpoint(filepath)
        Emit a concise checkpoint-saved message.

    logger.info(msg) / logger.warning(msg) / logger.debug(msg)
        Pass-through to the underlying Python logger for ad-hoc messages.

Usage
-----
    from src.utils.logger import TrainingLogger

    logger = TrainingLogger()

    # Inside the training loop
    logger.log_episode(
        episode=10,
        max_episodes=1000,
        reward=12345.0,
        loss=0.0421,
        epsilon=0.9048,
        steps=100,
        buffer_size=640,
        duration=0.23,
    )

    # After training completes
    logger.log_summary(
        total_episodes=1000,
        mean_reward=13200.5,
        best_reward=18750.0,
        final_epsilon=0.05,
        total_duration=182.4,
    )
"""

from __future__ import annotations

import logging
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default output directory for log files (created automatically)
_DEFAULT_LOG_DIR      = os.path.join("outputs", "logs")

# Default log filename — one log file per project, appended across runs
_DEFAULT_LOG_FILENAME = "training.log"

# Logger name — named to avoid polluting the root logger
_LOGGER_NAME          = "dqn_training"

# File log format — full timestamp + level + message
_FILE_FORMAT  = "%(asctime)s | %(levelname)-8s | %(message)s"

# Console format — compact; no timestamp or level noise
_CONSOLE_FORMAT = "%(message)s"


# ---------------------------------------------------------------------------
# TrainingLogger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Structured logger for the DQN Dynamic Pricing training pipeline.

    Simultaneously writes to a rotating log file and the console.
    All episode fields are formatted with consistent alignment so the
    log file is easy to grep, tail, and parse programmatically.

    Args:
        log_dir       (str):  Directory where the log file is created.
                              Created automatically if it does not exist.
                              Default: ``"outputs/logs"``.
        log_filename  (str):  Name of the log file.
                              Default: ``"training.log"``.
        console_level (int):  Logging level for the console handler.
                              Use ``logging.WARNING`` to silence routine
                              episode lines on the console.
                              Default: ``logging.INFO``.
        file_level    (int):  Logging level for the file handler.
                              Default: ``logging.DEBUG`` (captures everything).
    """

    def __init__(
        self,
        log_dir:       str = _DEFAULT_LOG_DIR,
        log_filename:  str = _DEFAULT_LOG_FILENAME,
        console_level: int = logging.INFO,
        file_level:    int = logging.DEBUG,
    ) -> None:
        self._log_dir  = log_dir
        self._log_path = os.path.join(log_dir, log_filename)

        # Create the output directory if it does not already exist
        os.makedirs(log_dir, exist_ok=True)

        # Obtain (or create) the named logger — avoids modifying root logger
        self._logger = logging.getLogger(_LOGGER_NAME)
        self._logger.setLevel(logging.DEBUG)   # capture all levels at logger

        # Guard — skip handler setup if already attached (idempotent)
        if not self._logger.handlers:
            self._attach_file_handler(file_level)
            self._attach_console_handler(console_level)

        # Emit a run-start separator to the log file so multiple training
        # runs are visually distinct when the log is reviewed later.
        self._write_run_header()

    # ------------------------------------------------------------------
    # Private setup helpers
    # ------------------------------------------------------------------

    def _attach_file_handler(self, level: int) -> None:
        """Attach a FileHandler that appends to the log file.

        Args:
            level (int): Minimum logging level for this handler.
        """
        fh = logging.FileHandler(self._log_path, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_FILE_FORMAT))
        self._logger.addHandler(fh)

    def _attach_console_handler(self, level: int) -> None:
        """Attach a StreamHandler that prints to stdout.

        Args:
            level (int): Minimum logging level for this handler.
        """
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        self._logger.addHandler(ch)

    def _write_run_header(self) -> None:
        """Write a timestamped separator to mark the start of a new run."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        # Use DEBUG so the header only appears in the file, not the console
        self._logger.debug(sep)
        self._logger.debug(f"  NEW TRAINING RUN  —  started at {now}")
        self._logger.debug(sep)

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode:     int,
        max_episodes: int,
        reward:      float,
        loss:        float,
        epsilon:     float,
        steps:       int,
        buffer_size: int,
        duration:    float,
    ) -> None:
        """Log a single training episode as one structured record.

        Fields are padded for consistent column alignment in the log file,
        making it easy to parse with ``grep``, ``awk``, or pandas.

        Args:
            episode      (int):   Current episode number (1-indexed).
            max_episodes (int):   Total number of training episodes.
            reward       (float): Total undiscounted reward for this episode.
            loss         (float): Mean TD (Bellman) loss over gradient steps.
                                  0.0 if the buffer was still warming up.
            epsilon      (float): Exploration rate at the end of this episode.
            steps        (int):   Number of environment steps taken.
            buffer_size  (int):   Current number of transitions in the buffer.
            duration     (float): Wall-clock seconds taken for this episode.
        """
        msg = (
            f"[Ep {episode:04d}/{max_episodes}] "
            f"reward={reward:>10.2f}  "
            f"loss={loss:>8.4f}  "
            f"eps={epsilon:.4f}  "
            f"steps={steps:>4d}  "
            f"buf={buffer_size:>6d}  "
            f"dur={duration:.2f}s"
        )
        self._logger.info(msg)

    def log_summary(
        self,
        total_episodes: int,
        mean_reward:    float,
        best_reward:    float,
        final_epsilon:  float,
        total_duration: float,
    ) -> None:
        """Log a multi-line summary block at the end of a training run.

        The summary is written at INFO level so it appears in both the
        log file and the console.

        Args:
            total_episodes  (int):   Total episodes completed.
            mean_reward     (float): Mean episode reward across all episodes.
            best_reward     (float): Best (maximum) episode reward achieved.
            final_epsilon   (float): Epsilon value at the end of training.
            total_duration  (float): Total wall-clock training time (seconds).
        """
        sep  = "=" * 60
        mins = total_duration / 60

        self._logger.info(sep)
        self._logger.info("  TRAINING COMPLETE")
        self._logger.info(sep)
        self._logger.info(f"  Total episodes   : {total_episodes}")
        self._logger.info(f"  Mean reward      : {mean_reward:>10.4f}")
        self._logger.info(f"  Best reward      : {best_reward:>10.4f}")
        self._logger.info(f"  Final epsilon    : {final_epsilon:.6f}")
        self._logger.info(f"  Total duration   : {total_duration:.1f}s  ({mins:.2f} min)")
        self._logger.info(sep)

    def log_checkpoint(self, filepath: str) -> None:
        """Log a concise checkpoint-saved notification.

        Args:
            filepath (str): Path of the checkpoint file that was saved.
        """
        self._logger.debug(f"  [ckpt] Saved → {filepath}")

    def log_config(self, cfg_dict: dict) -> None:
        """Log the training configuration as key-value pairs.

        Useful for auditing which hyperparameters produced a given result
        when reviewing the log file after the fact.

        Args:
            cfg_dict (dict): Dictionary of configuration values (e.g. from
                             ``dataclasses.asdict(cfg)``).
        """
        self._logger.debug("  Config:")
        for key, value in cfg_dict.items():
            self._logger.debug(f"    {key:<28}: {value}")

    # ------------------------------------------------------------------
    # Pass-through convenience methods
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        """Emit an INFO-level message.

        Args:
            msg (str): Message text.
        """
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Emit a WARNING-level message.

        Args:
            msg (str): Message text.
        """
        self._logger.warning(msg)

    def debug(self, msg: str) -> None:
        """Emit a DEBUG-level message (file only, not console by default).

        Args:
            msg (str): Message text.
        """
        self._logger.debug(msg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> str:
        """str: Absolute path of the log file being written."""
        return os.path.abspath(self._log_path)
