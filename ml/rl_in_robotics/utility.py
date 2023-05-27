# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): RL in robotics
#           Training in Gym

# Necessary packages
import random
from time import sleep

import gym


def gym_rl_tasks_with_seed(env, episodes, action):
    """
    Unifying all RL tasks by Gym toolkit with seeding
    """
    seed = 1
    random.seed(seed)
    env.seed(seed)

    reset_render_send_action_tasks(env, episodes, action)
    env.close()


def gym_rl_tasks(env, episodes, action):
    """
    Unifying all RL tasks by Gym toolkit
    """
    reset_render_send_action_tasks(env, episodes, action)
    env.close()


def reset_render_send_action_tasks(env, episodes, action):
    """
    Unifying RL tasks by Gym toolkit
    """
    for i in range(episodes):
        env.reset()
        reward_sum = 0

        while True:
            env.render()
            action = action
            state, reward, done, debug = env.step(action)
            reward_sum += reward
            sleep(.01)
            if done:
                print(f'Episode {i} reward: {reward_sum}')
                sleep(1)
                break


def init_environment(rl_problem):
    """
     Initialize the environment
    """
    env = gym.make(rl_problem)
    return env


def check_list_of_environments():
    """
    Get a list of all pre-installed environments of Gym Toolkit
    :return: A list of all pre-installed environments
    """
    for e in gym.envs.registry.all():
        print(e.id)
