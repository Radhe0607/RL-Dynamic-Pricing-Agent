"""
q_learning_agent.py
-------------------
Tabular Q-Learning agent for the Dynamic Pricing environment.

This agent maintains a Q-table — a 2-D array indexed by (state, action) —
and updates it using the standard temporal-difference (TD) learning rule
after every environment step.

The agent is used as a lightweight interpretable baseline.  Unlike the
DQN agent it requires no neural network or GPU; its Q-table can be
inspected directly to understand the learned policy.

All hyperparameters are injected through ``QLearningConfig`` (defined in
``src/config/dqn_config.py``), which is the single source of truth for
every tunable number in the project.  No magic numbers live in this file.

Temporal-difference update rule
---------------------------------
    TD target = r + γ · max_a' Q(s', a')
    Q(s, a)  ← Q(s, a) + α · (TD target − Q(s, a))

Where:
    α  = learning_rate      (step size; controls how fast old values are overwritten)
    γ  = discount_factor    (weight given to future rewards)

ε-greedy exploration
---------------------
    With probability ε  → choose a random action (explore)
    Otherwise           → choose argmax_a Q(s, a) (exploit)

    ε is decayed externally by the caller each episode:
        ε_{t+1} = max(ε_end, ε_t · ε_decay)

Usage
-----
    from src.config.dqn_config import QLearningConfig
    from src.agents.q_learning_agent import QLearningAgent

    cfg   = QLearningConfig()
    agent = QLearningAgent(cfg)

    # Inside training loop
    action = agent.select_action(state, epsilon)
    agent.update(state, action, reward, next_state)

    # Override a value without editing the config file
    cfg2  = QLearningConfig.from_config({"learning_rate": 0.05})
    agent = QLearningAgent(cfg2)
"""

from __future__ import annotations

import random

import numpy as np

from src.config.dqn_config import QLearningConfig


class QLearningAgent:
    """Tabular Q-Learning agent driven entirely by ``QLearningConfig``.

    The Q-table is initialised to zeros, representing an uninformed prior
    (equal expected return for every state-action pair).

    Args:
        cfg (QLearningConfig): Hyperparameter configuration object.
                               All learning rates, discount factors, and
                               environment dimensions come from here.
    """

    def __init__(self, cfg: QLearningConfig | None = None) -> None:
        # Use defaults when no config is supplied
        self.cfg = cfg if cfg is not None else QLearningConfig()

        # Q-table: rows = discrete states, columns = discrete actions
        # Initialised to 0 — equivalent to assuming zero future return
        self.q_table: np.ndarray = np.zeros(
            (self.cfg.state_size, self.cfg.action_size),
            dtype=np.float64,
        )

        # Expose commonly accessed hyperparameters as attributes for readability
        self.learning_rate:   float = self.cfg.learning_rate
        self.discount_factor: float = self.cfg.discount_factor

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int, epsilon: float) -> int:
        """Choose an action using an ε-greedy policy.

        Args:
            state   (int):   Discrete state index (row in Q-table).
            epsilon (float): Current exploration probability ∈ [0, 1].

        Returns:
            int: Index of the chosen action (column in Q-table).
        """
        if random.random() < epsilon:
            # Exploration — uniformly random action
            return random.randrange(self.cfg.action_size)

        # Exploitation — greedy action from Q-table
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # TD update
    # ------------------------------------------------------------------

    def update(
        self,
        state:      int,
        action:     int,
        reward:     float,
        next_state: int,
        done:       bool = False,
    ) -> float:
        """Apply one TD update step to the Q-table.

        The Bellman update target is:
            y = r  +  γ · max_a' Q(s', a') · (1 − done)

        Terminal transitions (done=True) have their bootstrapped next-state
        value zeroed out, so the agent learns only from the immediate reward.

        Args:
            state      (int):   Current discrete state index.
            action     (int):   Action taken in *state*.
            reward     (float): Scalar reward received from the environment.
            next_state (int):   Successor discrete state index.
            done       (bool):  Whether the episode ended after this step.

        Returns:
            float: The absolute TD error |y − Q(s, a)| — useful for logging.
        """
        # Best Q-value achievable from next_state (zero if terminal)
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))

        # TD target — what we wish Q(s, a) would equal
        td_target = reward + self.discount_factor * best_next_q

        # TD error — how far our current estimate is from the target
        td_error = td_target - self.q_table[state, action]

        # Update the Q-table entry
        self.q_table[state, action] += self.learning_rate * td_error

        return abs(td_error)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def best_action(self, state: int) -> int:
        """Return the greedy (ε=0) action for *state*.

        Args:
            state (int): Discrete state index.

        Returns:
            int: Action index with the highest Q-value.
        """
        return int(np.argmax(self.q_table[state]))

    def reset_table(self) -> None:
        """Re-initialise the Q-table to zeros.

        Useful when running multiple independent training trials from
        the same agent instance.
        """
        self.q_table[:] = 0.0

    def __repr__(self) -> str:
        return (
            f"QLearningAgent("
            f"states={self.cfg.state_size}, "
            f"actions={self.cfg.action_size}, "
            f"lr={self.cfg.learning_rate}, "
            f"γ={self.cfg.discount_factor})"
        )