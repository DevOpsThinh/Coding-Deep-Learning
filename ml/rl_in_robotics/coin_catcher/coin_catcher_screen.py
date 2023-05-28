# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Reinforcement Learning (RL): Coin Catcher RL problem
#    Ref: https://www.pygame.org/docs

import os

import pygame as game

# Setting up the colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


class CoinCatcherScreen:
    """
    Drawing the game screen
    """
    img_path = os.path.dirname(os.path.realpath(__file__)) + '/airplane.png'
    rct_size = 50

    def __init__(self, h=5, w=5) -> None:
        game.init()
        self.h = h
        self.w = w
        scr_size = ((w + 2) * self.rct_size, (h + 3) * self.rct_size)
        self.scr = game.display.set_mode(scr_size, 0, 32)
        self.img = game.image.load(CoinCatcherScreen.img_path)
        self.font = game.font.SysFont("arial", 48)
        game.display.set_caption('Catch Coins Game')
        super().__init__()

    def plus(self):
        self.scr.fill(GREEN)
        game.display.update()

    def update(self, display, plane_pos, total_score):
        self.scr.fill(WHITE)
        for i in range(len(display)):
            line = display[len(display) - 1 - i]
            for j in range(len(line)):
                p = line[j]
                if p > 0:
                    coord = ((j + 1) * self.rct_size, (i + 1) * self.rct_size)
                    self.scr.blit(self.font.render(str(p), True, BLACK), coord)

        self.scr.blit(self.font.render(f'Total: {total_score}', True, BLACK), (10, 10))
        self.scr.blit(self.img, (self.rct_size * plane_pos + 30, (self.h + 1) * self.rct_size))
        game.display.update()

    @classmethod
    def render(cls, display, plane_pos, total_score):
        scr = CoinCatcherScreen()
        scr.update(display, plane_pos, total_score)
