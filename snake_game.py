# snake game that uses dependency injection to control it
# so I can write a manual control mechanism or an AI can do it

import pygame
import random
from typing import List, Tuple, Any
from misc import Direction, Coordinate
from enum import Enum


class DeathReason(Enum):
    NONE = -1
    WALL = 0
    TAIL = 1
    LOOP = 2


class AbstractSnakeGame:
    def __init__(self, use_renderer, dimensions: Tuple[int, int] = (600, 400)):
        self.render = use_renderer
        self.snake_alive: bool = True
        self.snake: List[Coordinate] = []
        self.dimensions: Tuple[int, int] = dimensions
        self.display: pygame.Surface = None
        self.clock: pygame.time.Clock = None
        self.food_location: Coordinate = None
        self.block_size: int = 10
        self.frametime: int = 10
        self.move_direction: Direction = Direction.NONE
        self.death_reason: DeathReason = DeathReason.NONE

    def play(self):
        self.make_walls()

        if self.render:
            pygame.init()
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
            self.display = pygame.display.set_mode(self.dimensions)
        self.snake.append(Coordinate(((self.dimensions[0] // self.block_size) // 2) * self.block_size, ((self.dimensions[1] // self.block_size) // 2) * self.block_size))
        self.food_location = self.generate_food()

        while self.snake_alive:
            self.handle_events()
            self.update()
            if self.render:
                self.draw()
                self.clock.tick(self.frametime)

    def make_walls(self):
        self.walls = []
        for i in range(0, self.dimensions[0], self.block_size):
            self.walls.append(Coordinate(i, 0))
            self.walls.append(Coordinate(i, self.dimensions[1] - self.block_size))
        for i in range(0, self.dimensions[1], self.block_size):
            self.walls.append(Coordinate(0, i))
            self.walls.append(Coordinate(self.dimensions[0] - self.block_size, i))

    def handle_events(self):
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake_alive = False

        new_direction = self.get_action()
        if new_direction == Direction.NONE and self.move_direction == Direction.NONE:
            return

        if new_direction != Direction.NONE:
            self.move_direction = new_direction

        self.move_snake()

    def get_action(self) -> Direction:
        return NotImplemented

    def move_snake(self):
        vector = self.move_direction.to_vector()
        new_head = Coordinate(self.snake[0].x + vector[0] * self.block_size, self.snake[0].y + vector[1] * self.block_size)
        # if new head is out of bounds, move it to the other side
        # if new_head.x < 0:
        #     new_head.x = self.dimensions[0] - self.block_size
        # elif new_head.x >= self.dimensions[0]:
        #     new_head.x = 0
        # elif new_head.y < 0:
        #     new_head.y = self.dimensions[1] - self.block_size
        # elif new_head.y >= self.dimensions[1]:
        #     new_head.y = 0

        self.snake.insert(0, new_head)
        self.snake.pop()

    def update(self):
        self.check_collision()

        if self.food_eaten():
            self.food_location = self.generate_food()
            self.snake.append(self.snake[-1])

    def food_eaten(self) -> bool:
        return self.snake[0].x == self.food_location.x and self.snake[0].y == self.food_location.y

    def check_collision(self):
        for coord in self.snake[1:]:
            if self.snake[0].x == coord.x and self.snake[0].y == coord.y:
                self.snake_alive = False
                self.death_reason = DeathReason.TAIL
                break

        # temp check if head is out of bounds
        if self.snake[0].x < self.block_size or self.snake[0].x >= self.dimensions[0]-self.block_size or self.snake[0].y < self.block_size or self.snake[0].y >= self.dimensions[1]-self.block_size:
            self.snake_alive = False
            self.death_reason = DeathReason.WALL

    def generate_food(self) -> Coordinate:
        x = random.randint(1, (self.dimensions[0] - self.block_size*2)//self.block_size) * self.block_size
        y = random.randint(1, (self.dimensions[1] - self.block_size*2)//self.block_size) * self.block_size

        xy = Coordinate(x, y)
        if xy in self.snake:
            return self.generate_food()

        return xy

    def draw(self):
        self.display.fill((0, 0, 0))
        self.draw_walls()
        self.draw_food()
        self.draw_snake()
        # self.draw_score()
        pygame.display.update()

    def draw_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.display, (50, 50, 50), (wall.x, wall.y, self.block_size, self.block_size))

    def draw_snake(self):
        for i, segment in enumerate(self.snake):
            pygame.draw.rect(self.display, (255, 255, 255), (segment.x, segment.y, self.block_size, self.block_size))
            if i == 0:
                pygame.draw.rect(self.display, (255, 0, 0), (segment.x, segment.y, self.block_size, self.block_size))

    def draw_food(self):
        pygame.draw.rect(self.display, (0, 255, 0), (self.food_location.x, self.food_location.y, self.block_size, self.block_size))

    def draw_score(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        score = self.get_score()
        text = font.render(f"Score: {score}", True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (self.dimensions[0] // 2, 50)
        self.display.blit(text, textRect)

    def get_score(self) -> int:
        return len(self.snake) - 1