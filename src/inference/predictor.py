"""
predictor.py
------------
Trained Model Inference for the DQN Dynamic Pricing Agent.

This module is the **only** component needed to go from a saved checkpoint
to a live price recommendation.  It is deliberately **independent of the
training pipeline** — it does not import train.py, metrics.py, or any
evaluation module.  The only training-side artefact it consumes is the
``.pt`` checkpoint file written by ``src/utils/checkpointing.py``.

Design principles
-----------------
* **No training dependency** — this module can be deployed on its own
  alongside ``src/agents/dqn_agent.py``, ``src/config/dqn_config.py``,
  and the checkpoint file.  Nothing else is required.
* **Deterministic inference** — epsilon is always 0 (greedy); randomness
  is disabled via ``torch.no_grad()`` and ``agent.online_network.eval()``.
* **Typed results** — every prediction returns an ``InferenceResult``
  dataclass that bundles the action index, recommended price, all Q-values,
  and a confidence score.  This makes the output easy to log, serialise,
  or forward to a downstream service.
* **Human-readable display** — ``InferenceResult.display()`` prints a
  clearly formatted recommendation panel.

State vector (3 features, matches PricingEnvironment.observation_space)
------------------------------------------------------------------------
  Index 0 — current_price     : current price being offered (float)
  Index 1 — remaining_inventory : unsold units remaining   (float)
  Index 2 — remaining_days    : days left in booking horizon (float)

Action → Price mapping
-----------------------
5 discrete price levels evenly spaced between MIN_PRICE and MAX_PRICE:

    action 0 → £10   (minimum / deep discount)
    action 1 → £32.50
    action 2 → £55   (mid-range)
    action 3 → £77.50
    action 4 → £100  (maximum / premium)

    price = MIN_PRICE + action × (MAX_PRICE − MIN_PRICE) / (n_actions − 1)

Public API
----------
    InferenceResult
        Dataclass holding the full output of one prediction call.

    PricingPredictor
        Main inference class.  Wraps a loaded DQNAgent in eval mode.

        PricingPredictor.from_checkpoint(path, cfg) → PricingPredictor
            Class-method constructor — recommended entry point.

        predictor.predict(state) → InferenceResult
            Predict the optimal price for a single state observation.

        predictor.predict_batch(states) → list[InferenceResult]
            Predict optimal prices for a list of states.

Usage (Python)
--------------
    from src.inference import PricingPredictor

    # Load from checkpoint
    predictor = PricingPredictor.from_checkpoint("checkpoints/dqn_final.pt")

    # Single prediction
    result = predictor.predict([50.0, 80.0, 30.0])
    result.display()

    # Batch prediction
    states  = [[50.0, 80.0, 30.0], [80.0, 20.0, 5.0]]
    results = predictor.predict_batch(states)
    for r in results:
        print(r.recommended_price)

Usage (CLI)
-----------
    python -m src.inference.run_inference checkpoints/dqn_final.pt
    python -m src.inference.run_inference checkpoints/dqn_final.pt \\
           --price 60 --inventory 40 --days 7
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.config.dqn_config import DQNConfig
from src.demand.config.config import MAX_PRICE, MIN_PRICE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of discrete price actions — matches PricingEnvironment.action_space.n
_N_ACTIONS: int = 5

# All price levels — computed once at import time
_PRICE_LEVELS: List[float] = [
    round(MIN_PRICE + i * (MAX_PRICE - MIN_PRICE) / (_N_ACTIONS - 1), 4)
    for i in range(_N_ACTIONS)
]

# Human-readable action labels for display
_ACTION_LABELS: List[str] = [
    f"Deep Discount  (£{p:.2f})" if i == 0 else
    f"Low Price      (£{p:.2f})" if i == 1 else
    f"Mid Price      (£{p:.2f})" if i == 2 else
    f"High Price     (£{p:.2f})" if i == 3 else
    f"Premium        (£{p:.2f})"
    for i, p in enumerate(_PRICE_LEVELS)
]

# State feature names — used in the display panel
_STATE_FEATURE_NAMES = ["current_price", "remaining_inventory", "remaining_days"]


# ---------------------------------------------------------------------------
# InferenceResult — typed output container
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Complete output of one price inference call.

    All fields are populated by ``PricingPredictor.predict()`` and
    are read-only after construction.

    Attributes
    ----------
    state              : Input state observation as a numpy array.
    action             : Chosen action index (0 – n_actions − 1).
    recommended_price  : Price in £ corresponding to the chosen action.
    all_prices         : All available price levels, in action-index order.
    q_values           : Raw Q-value estimates for every action (numpy array).
    confidence         : Softmax probability of the chosen action.
                         Higher → the model is more certain this is optimal.
    action_label       : Human-readable label for the chosen action.
    checkpoint_path    : Path of the checkpoint that produced this prediction.
    """

    state:             np.ndarray
    action:            int
    recommended_price: float
    all_prices:        List[float]
    q_values:          np.ndarray
    confidence:        float
    action_label:      str
    checkpoint_path:   str = ""

    def display(self) -> None:
        """Print a formatted recommendation panel to stdout.

        The panel shows the input state, all Q-values with their price
        labels, and clearly highlights the recommended price.

        Example output::

            ╔══════════════════════════════════════════════════════╗
            ║       DQN Pricing Recommendation                     ║
            ╠══════════════════════════════════════════════════════╣
            ║  Input State                                         ║
            ║    current_price       :  50.00                      ║
            ║    remaining_inventory :  80.00                      ║
            ║    remaining_days      :  30.00                      ║
            ╠══════════════════════════════════════════════════════╣
            ║  Q-Values (all price options)                        ║
            ║    [0] Deep Discount  (£10.00)   Q = -0.0312         ║
            ║    [1] Low Price      (£32.50)   Q =  4.1205  ←      ║
            ║    [2] Mid Price      (£55.00)   Q =  3.8901         ║
            ║    [3] High Price     (£77.50)   Q =  2.1044         ║
            ║    [4] Premium        (£100.00)  Q =  0.9911         ║
            ╠══════════════════════════════════════════════════════╣
            ║  ★  RECOMMENDED PRICE :  £32.50                     ║
            ║     Confidence        :  38.21 %                     ║
            ╚══════════════════════════════════════════════════════╝
        """
        w   = 58
        sep = "═" * w

        def _line(content: str = "") -> None:
            print(f"║  {content:<{w - 4}}║")

        print(f"╔{sep}╗")
        print(f"║  {'DQN Dynamic Pricing — Recommendation':<{w - 4}}║")
        print(f"╠{sep}╣")

        # ── Input state ──────────────────────────────────────────────────
        _line("Input State")
        for name, val in zip(_STATE_FEATURE_NAMES, self.state):
            _line(f"    {name:<24}: {val:>8.2f}")

        print(f"╠{sep}╣")

        # ── Q-values ─────────────────────────────────────────────────────
        _line("Q-Values  (all price options)")
        for i, (label, qval) in enumerate(zip(_ACTION_LABELS, self.q_values)):
            marker = "  ★" if i == self.action else ""
            _line(f"    [{i}] {label:<28}  Q = {qval:>8.4f}{marker}")

        print(f"╠{sep}╣")

        # ── Recommendation ───────────────────────────────────────────────
        _line(f"★  RECOMMENDED PRICE  :  £{self.recommended_price:.2f}")
        _line(f"   Action index       :  {self.action}")
        _line(f"   Confidence         :  {self.confidence * 100:.2f} %")

        if self.checkpoint_path:
            ckpt_name = os.path.basename(self.checkpoint_path)
            _line(f"   Checkpoint         :  {ckpt_name}")

        print(f"╚{sep}╝")

    def as_dict(self) -> dict:
        """Serialise the result to a plain Python dict (JSON-ready).

        Returns:
            dict: All fields with numpy arrays converted to lists.
        """
        return {
            "action":            self.action,
            "recommended_price": self.recommended_price,
            "all_prices":        self.all_prices,
            "q_values":          self.q_values.tolist(),
            "confidence":        round(self.confidence, 6),
            "action_label":      self.action_label,
            "state":             self.state.tolist(),
            "checkpoint_path":   self.checkpoint_path,
        }


# ---------------------------------------------------------------------------
# PricingPredictor — main inference class
# ---------------------------------------------------------------------------

class PricingPredictor:
    """Wraps a trained ``DQNAgent`` for deterministic price inference.

    The predictor always runs with epsilon = 0 (greedy) and with
    ``torch.no_grad()`` enabled, so no gradient computation occurs
    and no stochasticity is introduced by the inference call.

    The class is deliberately decoupled from the training pipeline:
    it only needs the checkpoint file, the ``DQNConfig`` dimensions
    (``state_size``, ``action_size``, ``hidden_size``), and access to
    ``DQNAgent``.

    Args:
        agent           (DQNAgent):   Loaded and eval-mode agent.
        checkpoint_path (str):        Path of the loaded checkpoint.
        cfg             (DQNConfig):  Config used to construct the agent.
        n_actions       (int):        Number of discrete price actions.
    """

    def __init__(
        self,
        agent:           DQNAgent,
        checkpoint_path: str        = "",
        cfg:             Optional[DQNConfig] = None,
        n_actions:       int        = _N_ACTIONS,
    ) -> None:
        self._agent           = agent
        self._checkpoint_path = checkpoint_path
        self._cfg             = cfg or DQNConfig()
        self._n_actions       = n_actions

        # Pre-compute price levels for this action space size
        self._price_levels: List[float] = [
            round(MIN_PRICE + i * (MAX_PRICE - MIN_PRICE) / max(n_actions - 1, 1), 4)
            for i in range(n_actions)
        ]

        # Switch the online network to eval mode — disables dropout / batch-norm
        # noise and signals PyTorch that gradients are not needed.
        self._agent.online_network.eval()

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        cfg:             Optional[DQNConfig] = None,
    ) -> "PricingPredictor":
        """Load a trained checkpoint and return a ready-to-use predictor.

        This is the **recommended** way to instantiate ``PricingPredictor``.
        It reads the checkpoint, restores network weights, and puts the
        model in eval mode — all in one call.

        Args:
            checkpoint_path (str):            Path to a ``.pt`` checkpoint
                                              produced by ``save_checkpoint``.
            cfg (DQNConfig | None):           Config object that defines the
                                              network dimensions.  When
                                              ``None``, ``DQNConfig()``
                                              defaults are used, which match
                                              the standard training setup.

        Returns:
            PricingPredictor: Fully initialised predictor.

        Raises:
            FileNotFoundError: If *checkpoint_path* does not exist.
            RuntimeError:      If the checkpoint architecture is
                               incompatible with *cfg*.

        Example::

            predictor = PricingPredictor.from_checkpoint(
                "checkpoints/dqn_final.pt"
            )
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint_path}'. "
                "Run training first or provide a valid path."
            )

        cfg = cfg or DQNConfig()

        # Build a fresh agent (optimiser state not needed for inference)
        agent = DQNAgent.from_config(cfg)

        # Load the checkpoint — this restores weights for both networks
        data = torch.load(checkpoint_path, map_location=agent.device)

        # Support both full training checkpoints and bare weight files
        if "online_state_dict" in data:
            # Full checkpoint produced by save_checkpoint()
            agent.online_network.load_state_dict(data["online_state_dict"])
            agent.target_network.load_state_dict(
                data.get("target_state_dict", data["online_state_dict"])
            )
            episode = data.get("episode", "unknown")
            epsilon = data.get("epsilon", "unknown")
            print(f"  [inference] Loaded checkpoint ← {checkpoint_path}")
            print(f"  [inference] Saved at episode {episode}  |  ε={epsilon}")
        else:
            # Bare state_dict saved by DQNAgent.save()
            agent.online_network.load_state_dict(data)
            agent.target_network.load_state_dict(data)
            print(f"  [inference] Loaded weights ← {checkpoint_path}")

        return cls(
            agent=agent,
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            n_actions=cfg.action_size,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(
        self,
        state: Union[List[float], np.ndarray],
    ) -> InferenceResult:
        """Predict the optimal pricing action for a single state observation.

        The prediction is **fully deterministic**: epsilon = 0 and
        ``torch.no_grad()`` is active for the entire call.

        Args:
            state (list[float] | np.ndarray): Observation vector of length
                ``state_size`` (default 3):
                ``[current_price, remaining_inventory, remaining_days]``.

        Returns:
            InferenceResult: Structured result containing the recommended
                             price, Q-values, confidence, and display helper.

        Raises:
            ValueError: If *state* has the wrong number of features.

        Example::

            result = predictor.predict([50.0, 80.0, 30.0])
            print(f"Recommended price: £{result.recommended_price:.2f}")
        """
        state_arr = np.array(state, dtype=np.float32)

        # Validate shape
        expected = self._cfg.state_size
        if state_arr.ndim != 1 or len(state_arr) != expected:
            raise ValueError(
                f"State must be a 1-D array of length {expected}. "
                f"Got shape {state_arr.shape}."
            )

        # ── Forward pass ─────────────────────────────────────────────────
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(
            self._agent.device
        )  # shape: (1, state_size)

        with torch.no_grad():
            q_values_tensor = self._agent.online_network(state_tensor)
            # shape: (1, action_size)

        q_values = q_values_tensor.squeeze(0).cpu().numpy()  # shape: (action_size,)

        # ── Greedy action ─────────────────────────────────────────────────
        action = int(np.argmax(q_values))

        # ── Price conversion ──────────────────────────────────────────────
        recommended_price = self._price_levels[action]

        # ── Confidence = softmax probability of chosen action ─────────────
        # Subtracting the max before exp improves numerical stability
        exp_q      = np.exp(q_values - np.max(q_values))
        softmax    = exp_q / exp_q.sum()
        confidence = float(softmax[action])

        return InferenceResult(
            state=state_arr,
            action=action,
            recommended_price=recommended_price,
            all_prices=self._price_levels,
            q_values=q_values,
            confidence=confidence,
            action_label=_ACTION_LABELS[action] if action < len(_ACTION_LABELS)
                         else f"Action {action}",
            checkpoint_path=self._checkpoint_path,
        )

    def predict_batch(
        self,
        states: List[Union[List[float], np.ndarray]],
    ) -> List[InferenceResult]:
        """Predict optimal prices for a list of state observations.

        Each state is processed independently through the same greedy
        forward pass.  Results are returned in the same order as the
        input list.

        Args:
            states (list): Sequence of state observations.  Each element
                           must be a 1-D array-like of length ``state_size``.

        Returns:
            list[InferenceResult]: One result per input state.

        Example::

            states  = [[50.0, 80.0, 30.0], [80.0, 20.0, 5.0]]
            results = predictor.predict_batch(states)
            for r in results:
                r.display()
        """
        return [self.predict(s) for s in states]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def action_to_price(self, action: int) -> float:
        """Convert a raw action index to the corresponding price level.

        Args:
            action (int): Action index in ``[0, n_actions)``.

        Returns:
            float: Price in £.

        Raises:
            ValueError: If *action* is outside the valid range.
        """
        if not (0 <= action < self._n_actions):
            raise ValueError(
                f"Action {action} is out of range [0, {self._n_actions})."
            )
        return self._price_levels[action]

    @property
    def price_levels(self) -> List[float]:
        """list[float]: All available price levels in action-index order."""
        return list(self._price_levels)

    @property
    def checkpoint_path(self) -> str:
        """str: Path of the checkpoint this predictor was loaded from."""
        return self._checkpoint_path

    def __repr__(self) -> str:
        ckpt = os.path.basename(self._checkpoint_path) or "no checkpoint"
        return (
            f"PricingPredictor("
            f"state_size={self._cfg.state_size}, "
            f"n_actions={self._n_actions}, "
            f"prices={self._price_levels}, "
            f"checkpoint='{ckpt}')"
        )
