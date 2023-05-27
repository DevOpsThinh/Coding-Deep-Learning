# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): RL in robotics
#           Gym environments: Phoenix Reinforcement Learning
#                   This environment is one of the legendary Atari video game series.
#                   In this game, you have to maximize your score.

from ml.rl_in_robotics.utility import gym_rl_tasks, init_environment

# Hyperparameters we can adjust
EPISODES = 10

env = init_environment("Phoenix-v0")

random_action = env.action_space.sample()

gym_rl_tasks(env, EPISODES, random_action)
