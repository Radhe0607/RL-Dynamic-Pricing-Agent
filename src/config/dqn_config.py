"""
dqn_config.py
-------------
Centralised hyperparameter configuration for all RL agents in the
Dynamic Pricing project.

This module is the **single source of truth** for every tunable number
in the project.  Changing a value here propagates automatically to
the training loop, agent constructor, replay buffer, and evaluation
pipeline — no file needs to be edited twice.

Contents
---------
  DQNConfig          — all hyperparameters for the Deep Q-Network agent.
  QLearningConfig    — all hyperparameters for the tabular Q-learning agent.

Usage
-----
    from src.config.dqn_config import DQNConfig, QLearningConfig

    # DQN
    cfg = DQNConfig()
    print(cfg.learning_rate)       # 1e-3

    # Tabular Q-Learning
    ql_cfg = QLearningConfig()
    print(ql_cfg.learning_rate)    # 0.1

    # Override a single value without touching the rest
    fast_cfg = DQNConfig(max_episodes=200, learning_rate=5e-4)
"""

from dataclasses import dataclass


@dataclass
class DQNConfig:
    """
    All tunable hyperparameters for the DQN Dynamic Pricing Agent.

    Attributes
    ----------
    Environment
        state_size:         Number of features in the observation vector.
                            Matches PricingEnvironment.observation_space.shape[0].
        action_size:        Number of discrete price actions the agent can choose.
                            Matches PricingEnvironment.action_space.n.
        max_episodes:       Total training episodes to run.
        max_steps_per_episode: Hard cap on steps within a single episode.

    Replay Buffer
        buffer_capacity:    Maximum number of transitions stored in ReplayBuffer.
        batch_size:         Mini-batch size sampled per training step.
        min_buffer_size:    Minimum transitions in buffer before training starts.
                            Prevents learning from a near-empty, unrepresentative buffer.

    Network & Optimiser
        hidden_size:        Number of neurons in each hidden layer of the Q-network.
        learning_rate:      Adam optimiser step size.

    RL Core
        gamma:              Discount factor γ — how much the agent values future rewards.
                            γ=1 means fully far-sighted; γ=0 means myopic.
        tau:                Soft update coefficient for the target network.
                            target_weights = τ * online + (1−τ) * target.
        target_update_freq: Steps between hard target-network syncs
                            (used when tau == 1.0, i.e., hard updates).

    Exploration (ε-greedy)
        epsilon_start:      Initial exploration rate — agent acts randomly 100 % of the time.
        epsilon_end:        Minimum exploration rate — agent always explores at least this much.
        epsilon_decay:      Multiplicative decay applied to ε after every episode.
                            ε_{t+1} = max(ε_end, ε_t * ε_decay).

    Checkpointing
        checkpoint_dir:     Directory where model weights are saved.
        save_every:         Save a checkpoint every N episodes.
    """

    # ── Environment ────────────────────────────────────────────────────
    state_size: int = 3           # [price, inventory, demand] → 3 features
    action_size: int = 5          # 5 discrete price levels
    max_episodes: int = 1_000
    max_steps_per_episode: int = 200

    # ── Replay Buffer ───────────────────────────────────────────────────
    buffer_capacity: int = 10_000
    batch_size: int = 64
    min_buffer_size: int = 500    # warm-up transitions before first update

    # ── Network & Optimiser ─────────────────────────────────────────────
    hidden_size: int = 64
    learning_rate: float = 1e-3

    # ── RL Core ─────────────────────────────────────────────────────────
    gamma: float = 0.99           # strong preference for long-term revenue
    tau: float = 0.005            # soft update: blend 0.5 % of online weights
    target_update_freq: int = 100 # steps (used only when tau == 1.0)

    # ── Exploration ──────────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every: int = 100         # save model every N episodes

    def __post_init__(self) -> None:
        """Validate critical hyperparameter constraints at construction time."""
        assert 0.0 < self.gamma <= 1.0, "gamma must be in (0, 1]"
        assert 0.0 < self.tau <= 1.0,   "tau must be in (0, 1]"
        assert 0.0 < self.epsilon_end < self.epsilon_start <= 1.0, (
            "epsilon values must satisfy: 0 < epsilon_end < epsilon_start <= 1"
        )
        assert self.min_buffer_size >= self.batch_size, (
            "min_buffer_size must be >= batch_size to enable the first sample"
        )

    @classmethod
    def from_config(cls, overrides: dict | None = None) -> "DQNConfig":
        """Create a ``DQNConfig`` with optional field overrides.

        This factory makes it easy to construct a config programmatically
        (e.g. from a sweep script) without importing dataclasses directly.

        Args:
            overrides (dict | None): Key→value pairs to override.  Any key
                                     not in ``DQNConfig`` fields is silently
                                     ignored to keep callers forwards-compatible.

        Returns:
            DQNConfig: New instance with requested values overridden.

        Example::

            cfg = DQNConfig.from_config({"learning_rate": 5e-4, "gamma": 0.95})
        """
        import dataclasses
        if not overrides:
            return cls()
        valid = {f.name for f in dataclasses.fields(cls)}
        safe  = {k: v for k, v in overrides.items() if k in valid}
        return cls(**safe)


# ---------------------------------------------------------------------------
# Tabular Q-Learning configuration
# ---------------------------------------------------------------------------

@dataclass
class QLearningConfig:
    """
    All tunable hyperparameters for the tabular Q-Learning agent.

    The tabular agent is used as a lightweight baseline before the full
    DQN agent has been trained.  Its hyperparameters are structurally
    simpler — no neural network, no replay buffer — but are kept here
    alongside ``DQNConfig`` so that all numbers in the project live in
    one place.

    Attributes
    ----------
    Environment
        state_size:        Number of discrete states in the Q-table.
        action_size:       Number of discrete price actions.

    Learning
        learning_rate:     Step size α for Q-table updates.
                           Q(s,a) ← Q(s,a) + α·(TD target − Q(s,a)).
        discount_factor:   Discount factor γ ∈ (0, 1] for future rewards.

    Exploration (ε-greedy)
        epsilon_start:     Initial exploration rate (1.0 = fully random).
        epsilon_end:       Floor on the exploration rate.
        epsilon_decay:     Multiplicative decay per episode.
                           ε_{t+1} = max(ε_end, ε_t · ε_decay).

    Training
        max_episodes:      Total training episodes.
        max_steps_per_episode: Hard cap on steps within a single episode.
    """

    # ── Environment ────────────────────────────────────────────────────
    state_size:  int = 100    # number of discrete state buckets
    action_size: int = 5      # 5 discrete price levels (matches PricingEnvironment)

    # ── Learning ────────────────────────────────────────────────────────
    learning_rate:    float = 0.1    # α — Q-table update step size
    discount_factor:  float = 0.95   # γ — weight of future rewards

    # ── Exploration ──────────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end:   float = 0.05
    epsilon_decay: float = 0.995

    # ── Training ─────────────────────────────────────────────────────────
    max_episodes:          int = 1_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        """Validate hyperparameter constraints at construction time."""
        assert 0.0 < self.learning_rate <= 1.0, (
            "learning_rate must be in (0, 1]"
        )
        assert 0.0 < self.discount_factor <= 1.0, (
            "discount_factor must be in (0, 1]"
        )
        assert 0.0 < self.epsilon_end < self.epsilon_start <= 1.0, (
            "epsilon values must satisfy: 0 < epsilon_end < epsilon_start <= 1"
        )
        assert self.state_size >= 1, "state_size must be >= 1"
        assert self.action_size >= 2, "action_size must be >= 2"

    @classmethod
    def from_config(cls, overrides: dict | None = None) -> "QLearningConfig":
        """Create a ``QLearningConfig`` with optional field overrides.

        Args:
            overrides (dict | None): Key→value pairs to override.

        Returns:
            QLearningConfig: New instance with requested values overridden.

        Example::

            cfg = QLearningConfig.from_config({"learning_rate": 0.05, "max_episodes": 500})
        """
        import dataclasses
        if not overrides:
            return cls()
        valid = {f.name for f in dataclasses.fields(cls)}
        safe  = {k: v for k, v in overrides.items() if k in valid}
        return cls(**safe)
