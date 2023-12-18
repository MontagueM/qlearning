from typing import override

from misc import Direction
from snake_game import AbstractSnakeGame
import pygame


class KeyboardSnakeGame(AbstractSnakeGame):
    def __init__(self):
        super().__init__()
        self.frametime = 10

    @override
    def get_action(self) -> Direction:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return Direction.UP
        elif keys[pygame.K_DOWN]:
            return Direction.DOWN
        elif keys[pygame.K_LEFT]:
            return Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            return Direction.RIGHT
        else:
            return Direction.NONE


if __name__ == "__main__":
    game = KeyboardSnakeGame()
    game.play()