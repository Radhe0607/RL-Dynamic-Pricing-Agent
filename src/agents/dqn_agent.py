"""
dqn_agent.py
------------
Deep Q-Network (DQN) agent for the Dynamic Pricing environment.

All hyperparameters (learning rate, gamma, tau, hidden layer size, etc.)
are supplied through ``DQNConfig`` — defined in ``src/config/dqn_config.py``
and treated as the single source of truth for every tunable number.
No magic numbers live in this file.

Architecture
------------
Two identical Q-networks are maintained:
  - online_network  : trained every step via gradient descent.
  - target_network  : updated slowly via soft-copy (Polyak averaging).
                      Provides stable regression targets, preventing the
                      "moving target" problem that destabilises vanilla DQN.

Action Selection
----------------
ε-greedy policy: with probability ε the agent picks a random price action
(exploration); otherwise it picks the action with the highest predicted
Q-value (exploitation). ε is decayed externally by the training loop.

Reference: Mnih et al., "Human-level control through deep reinforcement
           learning", Nature 2015.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# DQNConfig is the single source of truth for all DQN hyperparameters.
# Import here so DQNAgent.from_config() can be used without the caller
# having to unpack individual fields.
from src.config.dqn_config import DQNConfig


# ---------------------------------------------------------------------------
# Q-Network definition (unchanged from skeleton — just documented)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Two-layer MLP that maps a state vector to Q-values for every action.

    Args:
        state_size  (int): Dimensionality of the observation space.
        action_size (int): Number of discrete actions (price levels).
        hidden_size (int): Width of both hidden layers.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns Q-values of shape (batch, action_size)."""
        return self.network(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN agent that wraps an online and a target Q-network.

    The agent handles:
      - ε-greedy action selection
      - Bellman-error (TD) loss computation
      - Gradient update of the online network
      - Soft update (Polyak averaging) of the target network

    Args:
        state_size  (int):   Observation vector length.
        action_size (int):   Number of discrete price actions.
        hidden_size (int):   Hidden layer width for both networks.
        learning_rate (float): Adam optimiser learning rate.
        gamma (float):       Discount factor γ ∈ (0, 1].
        tau   (float):       Soft-update rate τ ∈ (0, 1].
                             τ=1 performs a hard (full) copy.
        device (str):        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ) -> None:
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # ── Networks ────────────────────────────────────────────────────
        self.online_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)

        # Initialise target with the same weights; freeze gradient tracking
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # target is never trained directly

        # ── Optimiser & Loss ────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Choose an action using an ε-greedy policy.

        Args:
            state   (np.ndarray): Current environment observation (shape: state_size,).
            epsilon (float):      Exploration probability ∈ [0, 1].

        Returns:
            int: Index of the chosen action.
        """
        if random.random() < epsilon:
            # Exploration — random price level
            return random.randrange(self.action_size)

        # Exploitation — greedy action from the online network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_size)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)  # (1, action_size)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one gradient-descent step on a sampled mini-batch.

        Uses the standard DQN target:
            y = r  +  γ · max_a' Q_target(s', a')  ·  (1 − done)

        Args:
            states      (np.ndarray): shape (batch, state_size)
            actions     (np.ndarray): shape (batch,)  — int64
            rewards     (np.ndarray): shape (batch,)  — float32
            next_states (np.ndarray): shape (batch, state_size)
            dones       (np.ndarray): shape (batch,)  — float32 (0 or 1)

        Returns:
            float: Scalar loss value for logging.
        """
        # Convert numpy arrays to tensors on the correct device
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── Current Q-values  Q(s, a) ────────────────────────────────────
        # Gather the Q-value for the specific action that was actually taken
        current_q = self.online_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        # shape: (batch,)

        # ── Target Q-values  y = r + γ · max Q_target(s', ·) ────────────
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(dim=1).values
            # Zero out next-state value for terminal transitions
            target_q = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)
        # shape: (batch,)

        # ── Bellman (TD) loss ────────────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)

        # ── Gradient step ────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in early training
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def soft_update_target(self) -> None:
        """Polyak-average the target network towards the online network.

        θ_target ← τ · θ_online + (1 − τ) · θ_target

        A small τ (e.g. 0.005) keeps the target stable, providing
        consistent regression targets during training.
        """
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters(),
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    # Config-driven constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DQNConfig) -> "DQNAgent":
        """Construct a ``DQNAgent`` directly from a ``DQNConfig`` instance.

        This is the preferred construction path in the training pipeline:
        it guarantees that every hyperparameter comes from the central
        config object and that no value is accidentally hardcoded at the
        call site.

        Args:
            cfg (DQNConfig): Fully populated config object.

        Returns:
            DQNAgent: New agent instance configured according to *cfg*.

        Example::

            from src.config.dqn_config import DQNConfig
            from src.agents.dqn_agent import DQNAgent

            cfg   = DQNConfig(learning_rate=5e-4, gamma=0.95)
            agent = DQNAgent.from_config(cfg)
        """
        return cls(
            state_size=cfg.state_size,
            action_size=cfg.action_size,
            hidden_size=cfg.hidden_size,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            tau=cfg.tau,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save online network weights to disk.

        Args:
            path (str): File path for the checkpoint (e.g. 'checkpoints/ep100.pt').
        """
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: str) -> None:
        """Load online network weights from disk and sync the target network.

        Args:
            path (str): Path to a previously saved checkpoint.
        """
        self.online_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.online_network.state_dict())