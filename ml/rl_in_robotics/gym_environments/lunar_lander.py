# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): RL in robotics
#           Gym environments: Lunar Lander Reinforcement Learning
#                   The agent's goal is to land the lander between yellow flags.

from ml.rl_in_robotics.utility import gym_rl_tasks, init_environment

# Hyperparameters we can adjust
EPISODES = 10

env = init_environment("LunarLander-v2")

random_action = env.action_space.sample()

gym_rl_tasks(env, EPISODES, random_action)

