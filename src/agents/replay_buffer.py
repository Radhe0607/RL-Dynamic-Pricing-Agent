"""
replay_buffer.py
----------------
Experience Replay Buffer for DQN-based Dynamic Pricing Agent.

DQN requires a replay buffer to break temporal correlations between
consecutive transitions. Instead of learning from (s, a, r, s') pairs
as they arrive, the agent samples random mini-batches from this buffer,
which stabilises training and improves sample efficiency.

Reference: Mnih et al., "Human-level control through deep reinforcement
           learning", Nature 2015.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    """A single (state, action, reward, next_state, done) experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size circular buffer that stores experience transitions.

    When the buffer is full, the oldest transition is automatically
    discarded to make room for the newest one (FIFO via deque).

    Args:
        capacity (int): Maximum number of transitions to store.
        seed (int): Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self._buffer: deque = deque(maxlen=capacity)
        self._capacity: int = capacity
        random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition in the buffer.

        Args:
            state:      Observation before the action was taken.
            action:     Discrete action index chosen by the agent.
            reward:     Scalar reward received from the environment.
            next_state: Observation after the action was taken.
            done:       True if the episode terminated after this step.
        """
        transition = Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        )
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Randomly sample a mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample (must be ≤ len(self)).

        Returns:
            A tuple of stacked numpy arrays:
            (states, actions, rewards, next_states, dones)
            Each array has shape (batch_size, ...).

        Raises:
            ValueError: If the buffer has fewer transitions than batch_size.
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer "
                f"that contains only {len(self)} transitions."
            )

        batch: List[Transition] = random.sample(self._buffer, batch_size)

        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def is_ready(self, batch_size: int) -> bool:
        """Return True when enough transitions are stored to sample a batch.

        Args:
            batch_size: The batch size that will be requested during training.
        """
        return len(self) >= batch_size

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self._capacity}, "
            f"stored={len(self)})"
        )
