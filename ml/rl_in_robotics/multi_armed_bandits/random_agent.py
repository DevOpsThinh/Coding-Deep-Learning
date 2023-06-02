# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Multi-Armed Bandits RL problem

import numpy as np
import random as rad
from multiarmed_bandit_rl_problem import MultiArmedBanditEnv
from ml.rl_in_robotics.utility import gym_rl_custom_tasks

# Hyperparameters we can adjust
BANDITS = [.45, .45, .4, .6, .4]
SEED = 1

BALANCES = 1000
MODE = "ascii"  # "human"
SLEEP = .0          # ?

rad.seed(SEED)
np.random.seed(1)
env = MultiArmedBanditEnv(BANDITS)

# 5 one-armed bandits (0 -> 4)
action = rad.randint(0, 4)

gym_rl_custom_tasks(env, BALANCES, action, MODE, SLEEP)
