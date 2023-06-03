# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Multi-Armed Bandits RL problem

import numpy as np
import random as rad

from fundamentals.custom_functions import make_the_graph
from multiarmed_bandit_rl_problem import MultiArmedBanditEnv
from ml.rl_in_robotics.utility import run_a_policy

# Hyperparameters we can adjust
BANDITS = [.45, .45, .4, .6, .4]
SEED = 3  # 0 => Greedy Policy or 1 => Epsilon Greedy Policy or free to change it
POLICY = 3  # 1 => Greedy Policy or 2 => Epsilon Greedy Policy
BALANCES = 1000
EXPLORATION = 10
EPSILON = .1

MODE = "ascii"  # "human" - not available


rad.seed(SEED)
env, rewards = run_a_policy(MultiArmedBanditEnv(BANDITS), MODE, POLICY, BALANCES, EXPLORATION, EPSILON, visualize=True)
cum_rewards = np.cumsum(rewards)

# make_the_graph(cum_rewards, "Casino RL Problem with Epsilon Greedy Policy", "Trials", "Reward")
make_the_graph(cum_rewards, "Casino RL Problem with Thompson Sampling Policy", "Trials", "Reward")
