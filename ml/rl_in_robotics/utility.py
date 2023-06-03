# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): RL in robotics
#           Training in Gym

# Necessary packages
import random
from time import sleep
from scipy.stats import beta

import matplotlib.pyplot as plt
import numpy as np
import gym
import pygame


# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150


def run_a_policy(env, mode, policy, balance=1000, episodes=50, exploration=10, epsilon=.1, visualize=False):
    """
    Reinforcement Learning's Policies Algorithm In Action
    For our policy parameter case: https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
    """
    state = env.reset()
    rewards = []

    for e in range(balance):
        action = None

        match policy:
            case 1:
                action = greedy_policy(state, exploration)
            case 2:
                action = epsilon_greedy_policy(state, exploration, epsilon)
            case 3:
                if e % episodes == 0:
                    action = thompson_sampling_policy(state, visualize, plot_title=f'Iteration: {e}')
                else:
                    action = thompson_sampling_policy(state, False, plot_title=f'Iteration: {e}')

        state, reward, done, debug = env.step(action)
        rewards.append(reward)

    for _ in range(episodes):
        if mode == "human":
            env.render(mode)
        elif mode == "ascii":
            env.render("ascii")
        else:
            env.render()

    env.close()

    return env, rewards


def thompson_sampling_policy(state, visualize=True, plot_title=''):
    """
    Implement of the Thompson Sampling Policy
    """
    action = None
    max_bound = 0
    colors_list = ['red', 'blue', 'green', 'black', 'yellow']
    # Iterating each machine
    for m, trials in state.items():
        winner = len([r for r in trials if r == 1])
        loser = len([r for r in trials if r == -1])

        if winner + loser == 0:
            avg = 0
        else:
            avg = round(winner / (winner + loser), 2)
        # Generating random number for bandit's beta distributions:
        rad_beta = np.random.beta(winner + 1, loser + 1)
        if rad_beta > max_bound:
            max_bound = rad_beta
            action = m
        # Visualize
        if visualize:
            color = colors_list[m % len(colors_list)]
            x = np.linspace(beta.ppf(0.01, winner, loser), beta.ppf(0.99, winner, loser), 100)
            plt.plot(
                x, beta.ppf(x, winner, loser),
                label=f'Machine {m}| avg={avg}, v={round(rad_beta, 2)}',
                color=color,
                linewidth=3
            )
            plt.axvline(x=rad_beta, color=color, linestyle='--')

    if visualize:
        plt.title(f'Thompson Sampling: Beta Distribution - {plot_title}')
        plt.legend()
        plt.show()

    return action


def epsilon_greedy_policy(state, explore=10, epsilon=.1):
    """
    Implement of the Epsilon Greedy Policy
    """
    machines = len(state)
    trials = sum(len(state[m]) for m in range(machines))
    total_explore_trials = machines * explore

    # Exploration
    if trials <= total_explore_trials:
        return trials % machines
    # Random machine
    if random.random() < epsilon:
        return random.randint(0, machines - 1)
    # Exploitation
    avg_rewards = [sum(state[m]) / len(state[m]) for m in range(machines)]

    best_machine = np.argmax(avg_rewards)
    return best_machine


def greedy_policy(state, explore=10):
    """
    Implement of the Greedy Policy
    """
    machines = len(state)
    trials = sum(len(state[m]) for m in range(machines))
    total_explore_trials = machines * explore

    # Exploration
    if trials <= total_explore_trials:
        return trials % machines
    # Exploitation
    avg_rewards = [sum(state[m]) / len(state[m]) for m in range(machines)]

    best_machine = np.argmax(avg_rewards)
    return best_machine


def gym_rl_custom_tasks(env, episodes, action, mode, duration):
    """
    Unifying all RL custom tasks by Gym toolkit
    """
    init_reset_environment(env)

    gym_customize_tasks(env, episodes, action, mode, duration)

    env.close()


def init_reset_environment(env):
    env = env
    init_state = env.reset()
    return init_state


def gym_customize_tasks(env, episodes, action, mode="human", duration=1):
    """
    Unifying RL tasks by Gym toolkit
    """
    for _ in range(episodes):
        if mode == "human":
            env.render(mode)
        elif mode == "ascii":
            env.render("ascii")
        else:
            env.render()

        action = action
        # state, reward, done, debug = env.step(action)
        env.step(action)
        sleep(duration)


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


# Utilities function


def get_system_fonts():
    """
    Get all available fonts.
    :return: A list of all the fonts available on the system.
    """
    fonts = pygame.font.get_fonts()
    index = 1
    for font in fonts:
        print(f'{index}. {font},')
        index += 1


def check_list_of_environments():
    """
    Get a list of all pre-installed environments of Gym Toolkit
    :return: A list of all pre-installed environments
    """
    for e in gym.envs.registry.all():
        print(e.id)
