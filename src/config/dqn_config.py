"""
dqn_config.py
-------------
Centralised hyperparameter configuration for the DQN-based
Dynamic Pricing Agent.

Keeping all hyperparameters in one place makes it easy to run
experiments, reproduce results, and perform hyperparameter sweeps
without touching agent or training-loop code.

Usage
-----
    from src.config.dqn_config import DQNConfig
    cfg = DQNConfig()
    print(cfg.learning_rate)   # 1e-3
"""

from dataclasses import dataclass, field


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
