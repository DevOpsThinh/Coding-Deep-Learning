# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): RL in robotics
#           A classic example: Cart-Pole Reinforcement Learning
#           (https://www.youtube.com/watch?v=5Q14EjnOJZc)

# Necessary packages
import random
from time import sleep

import gym

"""
    The Gym simulates interaction with the CartPole environment 
"""

env = gym.make('CartPole-v1')

seed = 1
random.seed(seed)
env.seed(seed)

episodes = 10

for i in range(episodes):
    init_state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = 1 if init_state[2] > 0 else 0
        state, reward, done, debug = env.step(action)
        reward_sum += reward
        sleep(.01)
        if done:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(.01)
            if done:
                print(f'Episode {i} reward: {reward_sum}')
                sleep(1)
                break

env.close()
