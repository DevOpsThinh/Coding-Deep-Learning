# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): RL in robotics
#           Gym environments: Mountain Car Reinforcement Learning
#                   A car is positioned between two mountains.
#                   The goal is to reach the mountain on the right.

from ml.rl_in_robotics.utility import gym_rl_tasks, init_environment

# Hyperparameters we can adjust
EPISODES = 10

env = init_environment("MountainCar-v0")

random_action = env.action_space.sample()

gym_rl_tasks(env, EPISODES, random_action)
