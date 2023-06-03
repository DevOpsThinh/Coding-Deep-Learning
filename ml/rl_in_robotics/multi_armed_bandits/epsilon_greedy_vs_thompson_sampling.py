# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Multi-Armed Bandits RL problem

import numpy as np
import random as rad

import matplotlib.pyplot as plt

from multiarmed_bandit_rl_problem import MultiArmedBanditEnv
from ml.rl_in_robotics.utility import run_a_policy

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150

# Hyperparameters we can adjust
BANDITS = [.45, .45, .4, .6, .4]
SEED = 0
EPISODES = 50
EXPLORATION = 10
EPSILON = .1

MODE = "ascii"  # "human" - not available

rad.seed(SEED)
np.random.seed(SEED)

env = MultiArmedBanditEnv(BANDITS)
rewards = {
    'Epsilon Greedy': [np.cumsum(run_a_policy(env, MODE, policy=2, episodes=EPISODES, exploration=EXPLORATION,
                                              epsilon=EPSILON, visualize=True)[1]) for _ in range(EPISODES)],
    'Thompson Sampling': [np.cumsum(run_a_policy(env, MODE, policy=3, episodes=EPISODES, visualize=True)[1])
                          for _ in range(EPISODES)],
}

for p, r in rewards.items():
    plt.plot(np.average(r, axis=0), label=p)

plt.title("Battle")
plt.xlabel("Rounds")
plt.ylabel("Average Returns")
plt.legend()
plt.show()
