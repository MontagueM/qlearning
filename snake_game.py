# snake game that uses dependency injection to control it
# so I can write a manual control mechanism or an AI can do it

import pygame
import random
import math
import numpy as np
import time
from typing import List, Tuple, Callable
from collections import deque
from enum import Enum
from dataclasses import dataclass

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def opposite(self):
        if self == Direction.UP:
            return Direction.DOWN
        elif self == Direction.DOWN:
            return Direction.UP
        elif self == Direction.LEFT:
            return Direction.RIGHT
        elif self == Direction.RIGHT:
            return Direction.LEFT

    def to_vector(self):
        if self == Direction.UP:
            return (0, -1)
        elif self == Direction.DOWN:
            return (0, 1)
        elif self == Direction.LEFT:
            return (-1, 0)
        elif self == Direction.RIGHT:
            return (1, 0)

@dataclass
class Coordinate:
    x: int
    y: int


class SnakeGame:
    snake_alive: bool = True
    snake: List[Coordinate] = []
    dimensions: Tuple[int, int] = (600, 400)
    screen: pygame.Surface = None
    clock: pygame.time.Clock = None
    food_location: Coordinate = None
    block_size: int = 10
    frametime: int = 100
    food_eaten: bool = True

    def start(self):
        pygame.init()
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.dimensions)

        while self.snake_alive:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.frametime)

    def handle_events(self):
        pass

    def update(self):
        if self.food_eaten:
            self.food_location = self.generate_food()
            self.food_eaten = False

    def generate_food(self):
        x = random.randint(0, self.dimensions[0] - self.block_size)
        y = random.randint(0, self.dimensions[1] - self.block_size)
        return Coordinate(x, y)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_snake()
        self.draw_food()
        self.draw_score()
        pygame.display.flip()

    def draw_snake(self):
        for i, segment in enumerate(self.snake):
            pygame.draw.rect(self.screen, (255, 255, 255), (segment.x, segment.y, self.block_size, self.block_size))
            if i == 0:
                pygame.draw.rect(self.screen, (0, 255, 0), (segment.x, segment.y, self.block_size, self.block_size))

    def draw_food(self):
        pygame.draw.rect(self.screen, (0, 255, 0), (self.food_location.x, self.food_location.y, 10, 10))

    def draw_score(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        score = len(self.snake) - 1
        text = font.render(f"Score: {score}", True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (self.dimensions[0] // 2, 50)
        self.screen.blit(text, textRect)


if __name__ == "__main__":
    game = SnakeGame()
    game.start()