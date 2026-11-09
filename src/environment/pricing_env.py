"""Gymnasium environment for a finite-horizon dynamic pricing problem."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.demand.config.config import INITIAL_INVENTORY, MAX_PRICE, MIN_PRICE


class PricingEnvironment(gym.Env):
    """Simulate pricing fixed inventory over a booking horizon.

    Observation: current price, remaining inventory, and remaining periods.
    Actions select one of five evenly spaced prices. Demand decreases with price
    and includes seeded uniform market noise. Reward is realized sales revenue.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 100,
        initial_inventory: int = INITIAL_INVENTORY,
        demand_noise: float = 10.0,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if initial_inventory <= 0:
            raise ValueError("initial_inventory must be positive")
        if demand_noise < 0:
            raise ValueError("demand_noise cannot be negative")

        self.horizon = int(horizon)
        self.initial_inventory = int(initial_inventory)
        self.demand_noise = float(demand_noise)
        self.prices = np.linspace(MIN_PRICE, MAX_PRICE, 5, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.prices))
        self.observation_space = spaces.Box(
            low=np.array([MIN_PRICE, 0, 0], dtype=np.float32),
            high=np.array([MAX_PRICE, self.initial_inventory, self.horizon], dtype=np.float32),
            dtype=np.float32,
        )
        self._price = float(self.prices[len(self.prices) // 2])
        self._inventory = self.initial_inventory
        self._elapsed_steps = 0
        self._done = True

    def _observation(self) -> np.ndarray:
        return np.array(
            [self._price, self._inventory, self.horizon - self._elapsed_steps],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._price = float(self.prices[len(self.prices) // 2])
        self._inventory = self.initial_inventory
        self._elapsed_steps = 0
        self._done = False
        return self._observation(), {"inventory": self._inventory}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode is finished; call reset() before stepping again.")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.action_space.n - 1}]")

        self._price = float(self.prices[int(action)])
        market_noise = float(self.np_random.uniform(-self.demand_noise, self.demand_noise))
        potential_demand = max(0, int(round(100.0 - self._price + market_noise)))
        units_sold = min(potential_demand, self._inventory)
        revenue = self._price * units_sold

        self._inventory -= units_sold
        self._elapsed_steps += 1
        terminated = self._inventory <= 0 or self._elapsed_steps >= self.horizon
        truncated = False
        self._done = terminated

        info = {
            "price": self._price,
            "demand": potential_demand,
            "units_sold": units_sold,
            "revenue": revenue,
            "remaining_inventory": self._inventory,
        }
        return self._observation(), float(revenue), terminated, truncated, info
