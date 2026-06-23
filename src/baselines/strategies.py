"""
strategies.py
-------------
Baseline pricing strategies for the Dynamic Pricing environment.

Each strategy follows a common interface — a ``select_action(state)``
method that receives the current environment observation and returns a
discrete action index (0–4, matching ``PricingEnvironment.action_space``).

This allows every strategy (including the trained DQN agent) to be
evaluated through the same comparison harness in
``src/evaluation/comparison.py``.

Strategies implemented
-----------------------
FixedPriceStrategy
    Always selects the same price level.  Useful as the most
    conservative baseline — it represents the "do nothing" policy.

RandomPriceStrategy
    Selects a uniformly random action each step.  Represents a
    lower-bound benchmark; any meaningful learning should beat this.

RuleBasedPricingStrategy
    Adjusts the price level based on two signals read from the
    environment state:
      - Remaining inventory  (state[1])
      - Remaining time / days (state[2])
    When inventory is high relative to remaining time the strategy
    lowers prices to stimulate demand; when inventory is scarce it
    raises prices to maximise revenue per unit.

Price Level Mapping (action → price)
--------------------------------------
The environment's 5 discrete actions map to price points derived from
the ``MIN_PRICE`` and ``MAX_PRICE`` constants defined in
``src/demand/config/config.py``:

    action 0 → MIN_PRICE                 (lowest)
    action 1 → MIN_PRICE + 1 * step
    action 2 → MIN_PRICE + 2 * step      (mid)
    action 3 → MIN_PRICE + 3 * step
    action 4 → MAX_PRICE                 (highest)

where step = (MAX_PRICE − MIN_PRICE) / (n_actions − 1).

Usage
-----
    from src.baselines.strategies import (
        FixedPriceStrategy,
        RandomPriceStrategy,
        RuleBasedPricingStrategy,
    )

    strategy = RuleBasedPricingStrategy(n_actions=5)
    action   = strategy.select_action(state)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np

# Price range constants — kept consistent with the rest of the project
from src.demand.config.config import MAX_PRICE, MIN_PRICE


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Abstract interface that all pricing strategies must implement.

    Every concrete strategy exposes a single ``select_action`` method
    so they can be plugged into the comparison harness without changes.

    Args:
        n_actions (int): Number of discrete price actions in the
                         environment's action space. Default is 5.
        name      (str): Human-readable label used in comparison reports.
    """

    def __init__(self, n_actions: int = 5, name: str = "BaseStrategy") -> None:
        self.n_actions = n_actions
        self.name      = name

        # Pre-compute the price level for each action index
        self._price_levels: List[float] = self._build_price_levels()

    def _build_price_levels(self) -> List[float]:
        """Return evenly spaced price points between MIN_PRICE and MAX_PRICE.

        Returns:
            list[float]: Length ``n_actions``, from MIN_PRICE to MAX_PRICE.
        """
        step = (MAX_PRICE - MIN_PRICE) / max(self.n_actions - 1, 1)
        return [MIN_PRICE + i * step for i in range(self.n_actions)]

    def action_to_price(self, action: int) -> float:
        """Convert a discrete action index to its corresponding price.

        Args:
            action (int): Action index in ``[0, n_actions)``.

        Returns:
            float: Corresponding price level.
        """
        return self._price_levels[action]

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Choose a pricing action given the current environment state.

        Args:
            state (np.ndarray): Observation from the environment.
                                Shape ``(3,)`` = [price, inventory, days].

        Returns:
            int: Discrete action index in ``[0, n_actions)``.
        """

    def reset(self) -> None:
        """Optional hook called at the start of each evaluation episode.

        Override in subclasses that maintain episode-level internal state.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, n_actions={self.n_actions})"


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed Price
# ---------------------------------------------------------------------------

class FixedPriceStrategy(BaseStrategy):
    """Always selects the same price level throughout an episode.

    This is the most conservative baseline: it ignores all state
    information and applies one fixed action every step.  Any
    adaptive strategy should be expected to outperform this on
    revenue when demand varies over time.

    Args:
        action    (int):  Fixed action index to use every step.
                          Default is 2 (middle price level).
        n_actions (int):  Size of the discrete action space. Default is 5.
    """

    def __init__(self, action: int = 2, n_actions: int = 5) -> None:
        super().__init__(
            n_actions=n_actions,
            name=f"FixedPrice(action={action}, "
                 f"price={MIN_PRICE + action * (MAX_PRICE - MIN_PRICE) / max(n_actions - 1, 1):.0f})",
        )
        if not (0 <= action < n_actions):
            raise ValueError(
                f"action must be in [0, {n_actions}), got {action}."
            )
        self._fixed_action = action

    def select_action(self, state: np.ndarray) -> int:
        """Return the fixed action, ignoring the state completely.

        Args:
            state (np.ndarray): Current observation (ignored).

        Returns:
            int: The pre-configured fixed action index.
        """
        return self._fixed_action


# ---------------------------------------------------------------------------
# Strategy 2 — Random Price
# ---------------------------------------------------------------------------

class RandomPriceStrategy(BaseStrategy):
    """Selects a uniformly random action at every step.

    This is the lower-bound benchmark.  A well-trained DQN agent
    should consistently outperform random pricing on cumulative reward.

    Args:
        n_actions (int): Size of the discrete action space. Default is 5.
        seed      (int | None): Random seed for reproducibility.
                                Set to ``None`` for non-deterministic runs.
    """

    def __init__(self, n_actions: int = 5, seed: int | None = None) -> None:
        super().__init__(n_actions=n_actions, name="RandomPrice")
        self._rng = random.Random(seed)

    def select_action(self, state: np.ndarray) -> int:
        """Return a uniformly random action, ignoring the state.

        Args:
            state (np.ndarray): Current observation (ignored).

        Returns:
            int: Random action index in ``[0, n_actions)``.
        """
        return self._rng.randrange(self.n_actions)


# ---------------------------------------------------------------------------
# Strategy 3 — Rule-Based Pricing
# ---------------------------------------------------------------------------

class RuleBasedPricingStrategy(BaseStrategy):
    """Adjusts prices based on remaining inventory and remaining days.

    Intuition
    ----------
    Revenue management theory suggests that prices should be higher
    when supply is scarce and lower when there is excess inventory
    relative to remaining selling time.

    This strategy computes a simple **urgency ratio**:

        urgency = remaining_inventory / max(remaining_days, 1)

    High urgency (lots of inventory, few days left) → lower prices to
    move stock.  Low urgency (little inventory, many days left) → raise
    prices to maximise revenue per unit.

    State vector interpretation
    ----------------------------
    The environment's 3-dimensional observation is interpreted as:
      state[0] → current price  (not used for decision; may change)
      state[1] → remaining inventory
      state[2] → remaining days / time steps

    Action selection thresholds
    ----------------------------
    The urgency ratio is mapped to action levels via configurable
    thresholds ``high_threshold`` and ``low_threshold``:

        urgency > high_threshold → action 0 (lowest price — discount)
        urgency < low_threshold  → action 4 (highest price — premium)
        otherwise                → action 2 (mid price — neutral)

    Args:
        n_actions       (int):   Action space size. Default is 5.
        high_threshold  (float): Urgency above this → discount pricing.
                                 Default is 3.0.
        low_threshold   (float): Urgency below this → premium pricing.
                                 Default is 1.0.
    """

    def __init__(
        self,
        n_actions:      int   = 5,
        high_threshold: float = 3.0,
        low_threshold:  float = 1.0,
    ) -> None:
        super().__init__(
            n_actions=n_actions,
            name=f"RuleBased(hi={high_threshold}, lo={low_threshold})",
        )
        self._high = high_threshold
        self._low  = low_threshold

        # Pre-compute action indices for each regime
        self._action_discount = 0                        # lowest price
        self._action_premium  = n_actions - 1           # highest price
        self._action_neutral  = n_actions // 2          # mid price

    def select_action(self, state: np.ndarray) -> int:
        """Choose an action based on inventory urgency ratio.

        Args:
            state (np.ndarray): Observation ``[price, inventory, days]``.

        Returns:
            int: Action index in ``[0, n_actions)``.
        """
        # Extract the two signals that drive the decision
        inventory      = float(state[1]) if len(state) > 1 else 50.0
        remaining_days = float(state[2]) if len(state) > 2 else 10.0

        # Avoid division by zero; treat 0 remaining days as 1
        urgency = inventory / max(remaining_days, 1.0)

        if urgency > self._high:
            # Too much stock, not enough time → cut prices to clear inventory
            return self._action_discount
        elif urgency < self._low:
            # Scarce stock, plenty of time → charge a premium
            return self._action_premium
        else:
            # Balanced supply / demand → hold at mid price
            return self._action_neutral
