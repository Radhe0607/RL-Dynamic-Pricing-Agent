"""
dqn_config.py

Configuration settings for DQN training.
"""

# Learning parameters
LEARNING_RATE = 0.001
GAMMA = 0.99

# Replay buffer
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Target network
TARGET_UPDATE = 10

# Training
EPISODES = 500
MAX_STEPS = 200

# Saving
SAVE_MODEL_EVERY = 50