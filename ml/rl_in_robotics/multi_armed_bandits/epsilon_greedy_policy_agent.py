# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Multi-Armed Bandits RL problem

import numpy as np
import random as rad

from fundamentals.custom_functions import make_the_graph
from multiarmed_bandit_rl_problem import MultiArmedBanditEnv
from ml.rl_in_robotics.utility import run_epsilon_greedy_policy

# Hyperparameters we can adjust
BANDITS = [.45, .45, .4, .6, .4]
SEED = 1  # 0 when use greedy policy
POLICY = 2  # 1 when use epsilon greedy policy
BALANCES = 1000
EXPLORATION = 10
EPSILON = .1

MODE = "ascii"  # "human"


rad.seed(SEED)
env, rewards = run_epsilon_greedy_policy(MultiArmedBanditEnv(BANDITS), MODE, POLICY, BALANCES, EXPLORATION, EPSILON)
cum_rewards = np.cumsum(rewards)

make_the_graph(cum_rewards, "Casino RL Problem with Epsilon Greedy Policy", "Trials", "Reward")
