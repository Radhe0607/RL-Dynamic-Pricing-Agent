"""
checkpointing.py
----------------
Model checkpointing utilities for the DQN Dynamic Pricing Agent.

Why a dedicated module?
-----------------------
Saving *only* the network weights (state_dict) is sufficient for inference,
but resuming a training run correctly also requires:
  - The optimiser state  (momentum buffers, adaptive LR accumulators)
  - The current epsilon  (exploration schedule position)
  - The episode number   (so logs and filenames stay consistent)
  - Hyperparameter metadata (for auditing which config produced a checkpoint)

This module provides two public functions:
  save_checkpoint(agent, episode, epsilon, cfg, path)
  load_checkpoint(agent, path)            → resume_info dict

They can be called from any training loop without tight coupling to the
DQNAgent implementation details.

Usage
-----
    from src.utils.checkpointing import save_checkpoint, load_checkpoint

    # Save during training
    save_checkpoint(agent, episode=200, epsilon=0.42, cfg=cfg)

    # Resume a run
    info = load_checkpoint(agent, path="checkpoints/dqn_ep200.pt")
    start_episode = info["episode"] + 1
    epsilon       = info["epsilon"]
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch

# Type alias for the training-state dictionary stored inside a checkpoint file
CheckpointDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    agent: Any,
    episode: int,
    epsilon: float,
    cfg: Any,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Serialise the full training state of a DQNAgent to disk.

    The checkpoint file is a plain PyTorch `.pt` archive containing:
      - ``online_state_dict``  : weights of the online (trained) Q-network.
      - ``target_state_dict``  : weights of the frozen target Q-network.
      - ``optimizer_state_dict``: Adam moment accumulators & adaptive LR state.
      - ``episode``            : episode number at save time (for resuming).
      - ``epsilon``            : current exploration rate (for resuming decay).
      - ``timestamp``          : ISO-8601 UTC timestamp of the save.
      - ``config``             : copy of the DQNConfig fields as a plain dict.

    Args:
        agent     : A ``DQNAgent`` instance (must expose ``online_network``,
                    ``target_network``, and ``optimizer`` attributes).
        episode   (int):   The episode number at which this checkpoint is saved.
        epsilon   (float): Current ε value of the ε-greedy policy.
        cfg       : A ``DQNConfig`` dataclass instance (serialised to dict).
        directory (str, optional): Folder to write the checkpoint to.
                    Defaults to ``cfg.checkpoint_dir`` if not supplied.
        filename  (str, optional): Override the auto-generated filename.
                    Auto name format: ``dqn_ep{episode:05d}.pt``.

    Returns:
        str: Absolute path of the saved checkpoint file.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    # ── Resolve output path ──────────────────────────────────────────────
    save_dir = directory if directory is not None else cfg.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"dqn_ep{episode:05d}.pt"

    filepath = os.path.join(save_dir, filename)

    # ── Build the checkpoint dictionary ──────────────────────────────────
    checkpoint: CheckpointDict = {
        # Network weights
        "online_state_dict":    agent.online_network.state_dict(),
        "target_state_dict":    agent.target_network.state_dict(),
        # Optimiser state (momentum/adaptive-LR buffers)
        "optimizer_state_dict": agent.optimizer.state_dict(),
        # Training-loop state — required to resume correctly
        "episode":              episode,
        "epsilon":              epsilon,
        # Metadata — useful for auditing experiments
        "timestamp":            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config":               vars(cfg),          # dataclass → plain dict
    }

    torch.save(checkpoint, filepath)

    print(f"  [✓] Checkpoint saved  → {filepath}  (episode {episode}, ε={epsilon:.4f})")
    return filepath


def load_checkpoint(
    agent: Any,
    path: str,
    strict: bool = True,
) -> CheckpointDict:
    """Restore a DQNAgent's full training state from a checkpoint file.

    After calling this function the agent's networks and optimiser will
    reflect exactly the state they were in when ``save_checkpoint`` was
    called, making it possible to continue training from the saved episode.

    Args:
        agent      : A ``DQNAgent`` instance whose networks and optimiser
                     will be overwritten with the checkpoint data.
        path  (str): Path to the ``.pt`` checkpoint file produced by
                     ``save_checkpoint``.
        strict (bool): Whether to enforce strict key-matching when loading
                       ``state_dict`` into the networks. Set to ``False`` only
                       when fine-tuning on a modified architecture.

    Returns:
        dict: A ``CheckpointDict`` containing at least:
              - ``"episode"``  (int)   — episode number at save time.
              - ``"epsilon"``  (float) — ε value at save time.
              - ``"timestamp"`` (str)  — UTC ISO-8601 save timestamp.
              - ``"config"``   (dict)  — config fields at save time.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If the checkpoint is incompatible with the
                           current model architecture (when strict=True).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{path}'. "
            "Verify the path or run training from scratch."
        )

    # Load onto the same device the agent is already using
    checkpoint: CheckpointDict = torch.load(path, map_location=agent.device)

    # ── Restore network weights ──────────────────────────────────────────
    agent.online_network.load_state_dict(checkpoint["online_state_dict"], strict=strict)
    agent.target_network.load_state_dict(checkpoint["target_state_dict"], strict=strict)

    # ── Restore optimiser state ──────────────────────────────────────────
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Keep target network in eval mode (never trained directly)
    agent.target_network.eval()

    episode   = checkpoint.get("episode",   0)
    epsilon   = checkpoint.get("epsilon",   1.0)
    timestamp = checkpoint.get("timestamp", "unknown")

    print(f"  [✓] Checkpoint loaded ← {path}")
    print(f"       Saved at episode {episode}  |  ε={epsilon:.4f}  |  {timestamp}")

    return checkpoint


# ---------------------------------------------------------------------------
# Convenience: find the latest checkpoint in a directory
# ---------------------------------------------------------------------------

def latest_checkpoint(directory: str) -> Optional[str]:
    """Return the path of the most recently modified ``.pt`` file in *directory*.

    Useful for automatically resuming from the last saved checkpoint without
    needing to hard-code an episode number.

    Args:
        directory (str): Directory to search for checkpoint files.

    Returns:
        str | None: Absolute path of the newest ``.pt`` file, or ``None``
                    if the directory does not exist or contains no ``.pt`` files.
    """
    if not os.path.isdir(directory):
        return None

    pt_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pt")
    ]

    if not pt_files:
        return None

    # Sort by modification time — newest last
    return max(pt_files, key=os.path.getmtime)
