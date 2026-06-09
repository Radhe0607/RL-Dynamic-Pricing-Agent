import gymnasium as gym
from gymnasium import spaces


class PricingEnvironment(gym.Env):
    """
    RL Environment for Dynamic Pricing
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(3,),
            dtype=float
        )

    def reset(self):
        state = [50, 20, 10]
        return state, {}

    def step(self, action):
        reward = 0
        done = False
        state = [50, 20, 10]

        return state, reward, done, False, {}