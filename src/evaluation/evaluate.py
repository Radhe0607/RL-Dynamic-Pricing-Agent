"""
evaluate.py
-----------
Post-training evaluation utilities for the DQN Dynamic Pricing Agent.

This module is concerned with *what happened after training* — loading a
saved agent and measuring its greedy (ε=0) performance on the environment,
then printing a clean statistics report.

Separation of concerns
-----------------------
  metrics.py   — tracks stats *during* training (called by train.py)
  evaluate.py  — runs a *held-out* evaluation *after* training (this file)

Typical workflow
-----------------
    1. Train:    python -m src.training.train
    2. Evaluate: python -m src.evaluation.evaluate
                 (or import run_evaluation and call it from a notebook)
    3. Visualize: plots are saved automatically when save_plots=True
                  (see src/evaluation/visualization.py for standalone use)
    4. Compare:  python -m src.evaluation.comparison checkpoints/dqn_final.pt
                 (runs DQN vs Fixed / Random / Rule-Based baselines)

Usage
-----
    from src.evaluation.evaluate import run_evaluation
    from src.config.dqn_config import DQNConfig

    run_evaluation(
        checkpoint_path="checkpoints/dqn_final.pt",
        n_episodes=20,
    )
"""

from __future__ import annotations

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.config.dqn_config import DQNConfig
from src.environment.pricing_env import PricingEnvironment
from src.evaluation.visualization import plot_eval_rewards
from src.utils.checkpointing import load_checkpoint


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_greedy_episode(
    agent: DQNAgent,
    env: PricingEnvironment,
    max_steps: int,
) -> tuple[float, int]:
    """Roll out one episode with a fully greedy policy (ε = 0).

    Args:
        agent     (DQNAgent):          The loaded, trained agent.
        env       (PricingEnvironment): Environment instance.
        max_steps (int):               Hard cap on episode length.

    Returns:
        tuple[float, int]: (total_reward, steps_taken)
    """
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)

    total_reward = 0.0
    steps        = 0

    for _ in range(max_steps):
        # Pure exploitation — no exploration noise
        action = agent.select_action(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        steps        += 1
        state         = np.array(next_state, dtype=np.float32)

        if terminated or truncated:
            break

    return total_reward, steps


def _print_eval_report(
    rewards: list[float],
    steps: list[int],
    checkpoint_path: str,
) -> None:
    """Print a formatted evaluation statistics table to stdout.

    Args:
        rewards         (list[float]): Per-episode total rewards.
        steps           (list[int]):   Per-episode step counts.
        checkpoint_path (str):         Path of the checkpoint that was evaluated.
    """
    n = len(rewards)
    arr = np.array(rewards)

    separator = "=" * 58
    print(separator)
    print("  Post-Training Evaluation Report")
    print(separator)
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Episodes run : {n}")
    print(separator)
    print(f"  Mean reward  : {arr.mean():>10.3f}")
    print(f"  Std  reward  : {arr.std():>10.3f}")
    print(f"  Max  reward  : {arr.max():>10.3f}")
    print(f"  Min  reward  : {arr.min():>10.3f}")
    print(f"  Median reward: {float(np.median(arr)):>10.3f}")
    print(f"  Mean steps   : {np.mean(steps):>10.1f}")
    print(separator)

    # Per-episode breakdown
    print("  Episode breakdown:")
    print(f"  {'Ep':>4}  {'Reward':>10}  {'Steps':>6}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*6}")
    for i, (r, s) in enumerate(zip(rewards, steps), start=1):
        print(f"  {i:>4}  {r:>10.3f}  {s:>6}")
    print(separator)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    n_episodes: int = 10,
    cfg: DQNConfig | None = None,
    save_plots: bool = False,
    plot_dir: str = "outputs/plots",
) -> dict:
    """Evaluate a trained DQN agent over multiple greedy episodes.

    Loads the checkpoint, runs ``n_episodes`` fully greedy rollouts
    (ε = 0, no exploration), and prints a formatted statistics report.
    Optionally saves a reward bar-chart PNG via the visualization module.

    Args:
        checkpoint_path (str):          Path to a ``.pt`` checkpoint file
                                        produced by ``save_checkpoint``.
        n_episodes      (int):          Number of evaluation episodes to run.
                                        Default is 10.
        cfg             (DQNConfig | None): Config used to construct the agent.
                                        Defaults to ``DQNConfig()`` if omitted.
        save_plots      (bool):         When ``True``, saves an evaluation
                                        reward bar-chart to *plot_dir*.
                                        Default is ``False``.
        plot_dir        (str):          Directory where plot PNGs are written.
                                        Created automatically if missing.
                                        Default is ``"outputs/plots"``.

    Returns:
        dict: Evaluation results with keys:
              ``"rewards"`` (list[float]), ``"steps"`` (list[int]),
              ``"mean_reward"`` (float), ``"std_reward"`` (float).
    """
    if cfg is None:
        cfg = DQNConfig()

    # ── Construct agent and restore weights ──────────────────────────────
    agent = DQNAgent(
        state_size=cfg.state_size,
        action_size=cfg.action_size,
        hidden_size=cfg.hidden_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
    )
    load_checkpoint(agent, checkpoint_path)

    env = PricingEnvironment()

    rewards: list[float] = []
    steps:   list[int]   = []

    # ── Evaluation rollouts ──────────────────────────────────────────────
    for ep in range(n_episodes):
        ep_reward, ep_steps = _run_greedy_episode(
            agent,
            env,
            max_steps=cfg.max_steps_per_episode,
        )
        rewards.append(ep_reward)
        steps.append(ep_steps)

    # ── Report ─────────────────────────────────────────────────────────
    _print_eval_report(rewards, steps, checkpoint_path)

    arr = np.array(rewards)
    result = {
        "rewards":     rewards,
        "steps":       steps,
        "mean_reward": float(arr.mean()),
        "std_reward":  float(arr.std()),
    }

    # ── Optional: save evaluation reward plot ──────────────────────────────
    if save_plots:
        plot_eval_rewards(rewards, save_dir=plot_dir)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Allow passing a checkpoint path as a CLI argument:
    #   python -m src.evaluation.evaluate checkpoints/dqn_final.pt
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dqn_final.pt"
    run_evaluation(checkpoint_path=ckpt, n_episodes=10)
