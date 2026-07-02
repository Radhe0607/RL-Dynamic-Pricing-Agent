"""
train.py
--------
DQN Training Loop for the Dynamic Pricing Agent.

This script ties together every component built so far:
  ┌─────────────────────┐
  │  PricingEnvironment │  ← Gymnasium env (state, reward, done)
  └──────────┬──────────┘
             │ observations / rewards
  ┌──────────▼──────────┐
  │      DQNAgent       │  ← ε-greedy policy, Bellman loss, soft target update
  └──────────┬──────────┘
             │ (s, a, r, s', done) transitions
  ┌──────────▼──────────┐
  │    ReplayBuffer     │  ← fixed-capacity circular buffer, random sampling
  └─────────────────────┘
  Config read from DQNConfig (single source of truth for all hyperparameters).
  External overrides are loaded from config/training_config.json when present.

Training Loop (per episode)
---------------------------
1. Reset environment → get initial state.
2. For each step:
   a. Agent selects action with ε-greedy policy.
   b. Environment steps → (next_state, reward, done).
   c. Transition pushed into ReplayBuffer.
   d. If buffer has ≥ min_buffer_size transitions → sample batch → agent.learn().
   e. Soft-update target network.
3. Decay ε after the episode ends.
4. Log episode stats; save checkpoint every N episodes.

Configuration
-------------
  Priority order (highest wins):
    1. DQNConfig keyword arguments passed to train() directly.
    2. config/training_config.json  — loaded automatically when the file exists.
    3. DQNConfig dataclass defaults.

Usage
-----
    python -m src.training.train
    python -m src.training.train --config path/to/custom_config.json
"""

import os
import sys
import time
import dataclasses

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from src.config.config_loader import ConfigFileError, ConfigValidationError, load_config
from src.config.dqn_config import DQNConfig
from src.environment.pricing_env import PricingEnvironment
from src.evaluation.metrics import TrainingMetrics
from src.evaluation.report import generate_report
from src.utils.checkpointing import load_checkpoint, latest_checkpoint, save_checkpoint
from src.utils.logger import TrainingLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_agent(cfg: DQNConfig) -> DQNAgent:
    """Instantiate a ``DQNAgent`` from the given config.

    Delegates to ``DQNAgent.from_config()`` so all hyperparameters flow
    from the single centralised ``DQNConfig`` object — no value is
    repeated or hardcoded at the call site.

    Args:
        cfg (DQNConfig): Populated configuration object.

    Returns:
        DQNAgent: Ready-to-train agent.
    """
    return DQNAgent.from_config(cfg)


def decay_epsilon(epsilon: float, cfg: DQNConfig) -> float:
    """Apply multiplicative epsilon decay, floored at epsilon_end.

    ε_{t+1} = max(ε_end, ε_t * ε_decay)
    """
    return max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)


def ensure_checkpoint_dir(path: str) -> None:
    """Create the checkpoint directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train(
    cfg: DQNConfig | None = None,
    resume_from: str | None = None,
) -> None:
    """Run the full DQN training loop.

    Args:
        cfg (DQNConfig | None): Hyperparameter config. Defaults to DQNConfig()
                                if not provided, using all default values.
        resume_from (str | None): Path to a checkpoint ``.pt`` file produced
                                  by ``save_checkpoint``. When supplied, the
                                  agent's networks, optimiser, episode counter,
                                  and epsilon are restored before training begins.
                                  Pass ``"auto"`` to automatically resume from
                                  the most recent checkpoint in cfg.checkpoint_dir.
    """
    if cfg is None:
        cfg = DQNConfig()

    # ── Initialise components ────────────────────────────────────────────
    env    = PricingEnvironment()
    agent  = build_agent(cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
    ensure_checkpoint_dir(cfg.checkpoint_dir)

    epsilon      = cfg.epsilon_start   # exploration rate, annealed over training
    start_episode = 1                  # may be overridden when resuming

    # ── Optional: resume from a saved checkpoint ─────────────────────────
    if resume_from is not None:
        # Resolve "auto" → latest checkpoint file in the checkpoint directory
        if resume_from == "auto":
            resume_from = latest_checkpoint(cfg.checkpoint_dir)
            if resume_from is None:
                print("  [!] No checkpoints found in", cfg.checkpoint_dir, "— starting fresh.")

        if resume_from is not None:
            info         = load_checkpoint(agent, resume_from)
            start_episode = info.get("episode", 0) + 1   # resume from next episode
            epsilon       = info.get("epsilon", cfg.epsilon_start)
            print(f"  [→] Resuming from episode {start_episode}  (ε={epsilon:.4f})")

    print("=" * 60)
    print("  DQN Dynamic Pricing — Training Started")
    print("=" * 60)
    print(f"  Episodes       : {cfg.max_episodes}")
    print(f"  Steps/episode  : {cfg.max_steps_per_episode}")
    print(f"  Buffer capacity: {cfg.buffer_capacity}")
    print(f"  Batch size     : {cfg.batch_size}")
    print(f"  Warm-up steps  : {cfg.min_buffer_size}")
    print(f"  γ={cfg.gamma}  τ={cfg.tau}  lr={cfg.learning_rate}")
    print("=" * 60)

    # ── Initialise logger ────────────────────────────────────────────────
    logger = TrainingLogger()   # writes to outputs/logs/training.log
    logger.log_config(dataclasses.asdict(cfg))
    logger.info(f"  Training started  |  episodes={cfg.max_episodes}  "
                f"lr={cfg.learning_rate}  γ={cfg.gamma}  τ={cfg.tau}")

    # Metrics tracker — owns all reward/loss/epsilon history and formatting
    metrics = TrainingMetrics(window=50)

    start_time = time.time()

    for episode in range(start_episode, cfg.max_episodes + 1):

        # ── Reset ──────────────────────────────────────────────────────────
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0.0
        episode_loss   = 0.0
        steps_trained  = 0
        steps_taken    = 0   # total env steps this episode (for metrics)
        ep_start_time  = time.time()  # wall-clock start of this episode

        # ── Episode rollout ────────────────────────────────────────────────
        for step in range(cfg.max_steps_per_episode):

            # 1. Select action (ε-greedy)
            action = agent.select_action(state, epsilon)

            # 2. Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated

            # 3. Store transition in replay buffer
            buffer.push(state, action, reward, next_state, done)

            episode_reward += reward
            steps_taken    += 1
            state = next_state

            # 4. Learn — only once the buffer has enough transitions
            if buffer.is_ready(cfg.batch_size):
                states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)
                loss = agent.learn(states, actions, rewards, next_states, dones)
                episode_loss += loss
                steps_trained += 1

                # 5. Soft-update target network every step
                agent.soft_update_target()

            if done:
                break

        # ── Post-episode bookkeeping ───────────────────────────────────────────
        epsilon  = decay_epsilon(epsilon, cfg)
        avg_loss = episode_loss / steps_trained if steps_trained > 0 else 0.0

        # Record all signals into the metrics tracker
        metrics.record(
            episode=episode,
            reward=episode_reward,
            loss=avg_loss,
            epsilon=epsilon,
            steps=steps_taken,
            buffer_size=len(buffer),
        )

        # ── Logging (every 10 episodes) ─────────────────────────────────────
        if episode % 10 == 0 or episode == 1:
            metrics.print_progress(episode, cfg.max_episodes)

        # ── Structured file + console log (every episode) ──────────────────
        ep_duration = time.time() - ep_start_time
        logger.log_episode(
            episode=episode,
            max_episodes=cfg.max_episodes,
            reward=episode_reward,
            loss=avg_loss,
            epsilon=epsilon,
            steps=steps_taken,
            buffer_size=len(buffer),
            duration=ep_duration,
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if episode % cfg.save_every == 0:
            ckpt_path = save_checkpoint(
                agent=agent,
                episode=episode,
                epsilon=epsilon,
                cfg=cfg,
            )
            logger.log_checkpoint(ckpt_path)

    # ── Training complete ─────────────────────────────────────────────────────
    metrics.print_summary()

    # Emit final log summary
    history      = metrics.as_dict()
    all_rewards  = history.get("episode_rewards", [])
    logger.log_summary(
        total_episodes=cfg.max_episodes,
        mean_reward=float(sum(all_rewards) / max(len(all_rewards), 1)),
        best_reward=float(max(all_rewards)) if all_rewards else 0.0,
        final_epsilon=epsilon,
        total_duration=time.time() - start_time,
    )

    # Save text report summarising this training run
    generate_report(metrics, cfg)

    # Save final model with full training-state metadata
    save_checkpoint(
        agent=agent,
        episode=cfg.max_episodes,
        epsilon=epsilon,
        cfg=cfg,
        filename="dqn_final.pt",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Default path for the external JSON config (relative to project root)
_DEFAULT_CONFIG_PATH = "config/training_config.json"


if __name__ == "__main__":
    # ── Parse optional --config CLI argument ────────────────────────────
    # Usage:
    #   python -m src.training.train
    #   python -m src.training.train --config path/to/my_config.json
    #   python -m src.training.train --resume

    config_path = _DEFAULT_CONFIG_PATH
    resume      = False

    args = sys.argv[1:]
    i    = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] == "--resume":
            resume = True
            i += 1
        else:
            i += 1

    # ── Load config: external JSON → DQNConfig (falls back to defaults) ──
    if os.path.isfile(config_path):
        try:
            config = load_config(config_path)
        except ConfigFileError as exc:
            print(f"\n  [ERROR] Config file error: {exc}")
            sys.exit(1)
        except ConfigValidationError as exc:
            print(f"\n  [ERROR] Config validation failed: {exc}")
            sys.exit(1)
    else:
        print(
            f"  [config] No external config found at '{config_path}' "
            f"— using DQNConfig defaults."
        )
        config = DQNConfig()

    # ── Run training ───────────────────────────────────────────────
    train(config, resume_from="auto" if resume else None)
