import numpy as np


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95

    def update(self, state, action, reward, next_state):
        best_next_action = np.max(self.q_table[next_state])

        self.q_table[state, action] = (
            self.q_table[state, action]
            + self.learning_rate
            * (
                reward
                + self.discount_factor * best_next_action
                - self.q_table[state, action]
            )
        )