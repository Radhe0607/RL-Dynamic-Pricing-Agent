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

Usage
-----
    python -m src.training.train
"""

import os
import time

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from src.config.dqn_config import DQNConfig
from src.environment.pricing_env import PricingEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_agent(cfg: DQNConfig) -> DQNAgent:
    """Instantiate a DQNAgent from the given config."""
    return DQNAgent(
        state_size=cfg.state_size,
        action_size=cfg.action_size,
        hidden_size=cfg.hidden_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
    )


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

def train(cfg: DQNConfig | None = None) -> None:
    """Run the full DQN training loop.

    Args:
        cfg (DQNConfig | None): Hyperparameter config. Defaults to DQNConfig()
                                if not provided, using all default values.
    """
    if cfg is None:
        cfg = DQNConfig()

    # ── Initialise components ────────────────────────────────────────────
    env    = PricingEnvironment()
    agent  = build_agent(cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
    ensure_checkpoint_dir(cfg.checkpoint_dir)

    epsilon = cfg.epsilon_start   # exploration rate, annealed over training

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

    # Track training metrics for logging
    episode_rewards = []

    start_time = time.time()

    for episode in range(1, cfg.max_episodes + 1):

        # ── Reset ──────────────────────────────────────────────────────────
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0.0
        episode_loss   = 0.0
        steps_trained  = 0

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

        # ── Post-episode bookkeeping ───────────────────────────────────────
        epsilon = decay_epsilon(epsilon, cfg)
        episode_rewards.append(episode_reward)

        # Rolling average reward over last 50 episodes
        avg_reward = np.mean(episode_rewards[-50:])
        avg_loss   = episode_loss / steps_trained if steps_trained > 0 else 0.0

        # ── Logging (every 10 episodes) ────────────────────────────────────
        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            print(
                f"  Ep {episode:>5}/{cfg.max_episodes}"
                f"  |  reward={episode_reward:>8.2f}"
                f"  |  avg50={avg_reward:>8.2f}"
                f"  |  loss={avg_loss:.4f}"
                f"  |  ε={epsilon:.3f}"
                f"  |  buf={len(buffer):>6}"
                f"  |  {elapsed:>6.1f}s"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if episode % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"dqn_ep{episode}.pt")
            agent.save(ckpt_path)
            print(f"  [✓] Checkpoint saved → {ckpt_path}")

    # ── Training complete ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    final_avg  = np.mean(episode_rewards[-50:])

    print("=" * 60)
    print("  Training Complete")
    print(f"  Total time      : {total_time:.1f}s")
    print(f"  Final avg reward: {final_avg:.2f}  (last 50 episodes)")
    print("=" * 60)

    # Save final model
    final_path = os.path.join(cfg.checkpoint_dir, "dqn_final.pt")
    agent.save(final_path)
    print(f"  [✓] Final model saved → {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Override any default hyperparameters here before calling train()
    config = DQNConfig(
        max_episodes=500,          # quick smoke-test; increase for real training
        max_steps_per_episode=100,
    )
    train(config)
