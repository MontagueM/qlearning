# snake game that uses dependency injection to control it
# so I can write a manual control mechanism or an AI can do it

import pygame
import random
from typing import List, Tuple, Any
from misc import Direction, Coordinate


class AbstractSnakeGame:
    def __init__(self):
        self.snake_alive: bool = True
        self.snake: List[Coordinate] = []
        self.dimensions: Tuple[int, int] = (600, 400)
        self.display: pygame.Surface = None
        self.clock: pygame.time.Clock = None
        self.food_location: Coordinate = None
        self.block_size: int = 20
        self.frametime: int = 10
        self.move_direction: Direction = Direction.NONE

    def play(self):
        pygame.init()
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.dimensions)
        self.snake.append(Coordinate(self.dimensions[0] // 2, self.dimensions[1] // 2))
        self.food_location = self.generate_food()

        while self.snake_alive:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.frametime)

    def handle_events(self):
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
                break

        # temp check if head is out of bounds
        if self.snake[0].x < 0 or self.snake[0].x >= self.dimensions[0] or self.snake[0].y < 0 or self.snake[0].y >= self.dimensions[1]:
            self.snake_alive = False

    def generate_food(self):
        x = random.randint(0, (self.dimensions[0] - self.block_size)//self.block_size) * self.block_size
        y = random.randint(0, (self.dimensions[1] - self.block_size)//self.block_size) * self.block_size
        return Coordinate(x, y)

    def draw(self):
        self.display.fill((0, 0, 0))
        self.draw_snake()
        self.draw_food()
        self.draw_score()
        pygame.display.update()

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