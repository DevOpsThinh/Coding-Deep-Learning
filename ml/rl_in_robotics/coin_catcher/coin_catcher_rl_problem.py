# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Coin Catcher RL problem
import collections
import sys
from math import ceil
from random import random, randint
from time import sleep

import gym

from coin_catcher_screen import CoinCatcherScreen


class CoinCatcherEnv(gym.Env):
    """
     Coin Catcher environment in Gym
    """
    metadata = {"render.modes": ["human", "ascii"]}

    def __init__(self, display_width=10, display_height=10, density=.8):
        self.display_width = display_width
        self.display_height = display_height
        self.density = density
        self.display = collections.deque(maxlen=display_height)
        self.last_action = None
        self.last_reward = None
        self.total_score = 0
        self.v_position = 0
        self.game_scr = None

    def step(self, action):
        self.last_action = action
        self.v_position = min(max(self.v_position + action, 0), self.display_width - 1)
        reward = self.display[0][self.v_position]
        self.last_reward = reward
        self.total_score += reward
        self.display.append(self.line_generator())
        state = self.display, self.v_position
        done = False
        info = {}

        return state, reward, done, info

    def _render_human(self):
        """
         The Pygame's Graphic environment rendering
        """
        if not self.game_scr:
            self.game_scr = CoinCatcherScreen(
                h=self.display_height,
                w=self.display_width
            )

        if self.last_reward:
            self.game_scr.plus()
            sleep(.1)

        self.game_scr.update(
            self.display,
            self.v_position,
            self.total_score
        )

    def _render_ascii(self):
        """
         The ASCII environment rendering
        """
        outfile = sys.stdout
        area = []
        for i in range(self.display_height):
            line = self.display[self.display_height - 1 - i]
            row = []
            for j in range(len(line)):
                p = line[j]
                if p > 0:
                    row.append(str(p))
                    if i > 0 and area[-1][j] == ' ':
                        area[-1][j] = '|'
                    if i > 1 and area[-2][j] == ' ':
                        area[-2][j] = '.'
                else:
                    row.append(' ')

            area.append(row)

        pos_line = (['_'] * self.display_width)
        pos_line[self.v_position] = str(self.last_reward) if self.last_reward else 'V'

        area.append(pos_line)
        outfile.write(f"\nTotal score: {self.total_score} \n")
        outfile.write("\n".join("  ".join(line) for line in area) + "\n")

    def render(self, mode="ascii"):
        if mode == "human":
            self._render_human()
        elif mode == "ascii":
            self._render_ascii()
        else:
            raise Exception("Not Implemented!")

    def line_generator(self):
        line = [0] * self.display_width
        if random() > (1 - self.density):
            ran = random()
            if ran < .6:
                v = 1
            elif ran < .9:
                v = 2
            else:
                v = 3

            line[randint(0, self.display_width - 1)] = v
        return line

    def reset(self):
        for _ in range(self.display_height):
            self.display.append(self.line_generator())

        self.v_position = ceil(self.display_width / 2)
        state = self.display, self.v_position
        return state
