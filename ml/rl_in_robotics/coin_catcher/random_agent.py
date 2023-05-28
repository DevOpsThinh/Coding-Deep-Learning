# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Coin Catcher RL problem

import random
from coin_catcher_rl_problem import CoinCatcherEnv
from ml.rl_in_robotics.utility import gym_rl_custom_tasks

# Hyperparameters we can adjust
EPISODES = 1000
MODE = "human"  # "ascii"
SLEEP = .3          # 1

env = CoinCatcherEnv()

# -1: left
#  0: stay
# +1: right
action_space = [-1, 0, 1]
action = random.choice(action_space)

gym_rl_custom_tasks(env, EPISODES, action, MODE, SLEEP)
