# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Multi-Armed Bandits RL problem

import random
import gym


class MultiArmedBanditEnv(gym.Env):
    """
    A Multi-Armed Bandits gym's custom environment that simulates our casino problem.
    """
    metadata = {"render.modes": ["human", "ascii"]}

    def __init__(self, bandits):
        """
        Environment Initialization
        :param bandits: The winning probabilities
        """
        self.bandits = bandits
        self.state = {}
        self.reset()

    def step(self, action):
        """
        Takes the agent's action & calculates the rewards.
        Each action environment returns 1$ or -1$ to an agent
        :param action: The Agent's action
        :return: 1$ or -1$ to an agent.
        """
        p = self.bandits[action]
        r = random.random()
        reward = 1 if r <= p else -1
        self.state[action].append(reward)
        done = False
        debug = None

        return self.state, reward, done, debug

    def _render_human(self):
        """
         The Graphic environment rendering
        """

    def _render_ascii(self):
        """
         The ASCII environment rendering
        """
        returns = {}
        trials = {}

        for e in range(len(self.bandits)):
            returns[e] = sum(self.state[e])
            trials[e] = len(self.state[e])

        print(f'***** Total Trials: {sum(trials.values())} *****')

        for b, r in returns.items():
            t = trials[b]
            print(f'"Bandit {b}"| returns: {r}, trials: {t}')

        print(f'***** Total Returns: {sum(returns.values())} *****')

    def render(self, mode="human"):
        """
        Render the current state of the environment:
        Shows the overall statistics of all rounds
        """
        if mode == "human":
            self._render_human()
        elif mode == "ascii":
            self._render_ascii()
        else:
            raise Exception("Not Implemented!")

    def reset(self):
        """
        Reset environment to its original state
        :return:
        """
        self.state = {}

        for e in range(len(self.bandits)):
            self.state[e] = []

        return self.state
