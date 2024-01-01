"""
simpler version of snake game where only one cube and one food always in same location
used to prove out models with one of the simplest possible games
"""
import pygame
from typing import Tuple

from general import Agent, AbstractGame
from misc import Direction, Coordinate


class KeyboardAgent(Agent):
    def __init__(self):
        super().__init__(4)

    def act(self, game: AbstractGame):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            game.move_direction = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            game.move_direction = Direction.RIGHT
        elif keys[pygame.K_UP]:
            game.move_direction = Direction.UP
        elif keys[pygame.K_DOWN]:
            game.move_direction = Direction.DOWN
        else:
            game.move_direction = Direction.NONE

        game.act()

    def start_episode(self, episode_num):
        pass


class ExitGame(AbstractGame):
    def __init__(self, agent: Agent, use_renderer, dimensions: Tuple[int, int] = (100, 100)):
        self.render = use_renderer
        self.agent_position: Coordinate = None
        self.dimensions: Tuple[int, int] = dimensions
        self.display: pygame.Surface = None
        self.clock: pygame.time.Clock = None
        self.food_location: Coordinate = None
        self.block_size: int = 10
        self.frametime: int = 10
        self.move_direction: Direction = Direction.NONE

        self.agent: Agent = agent

        self.at_exit = False
        self.walls = []
        self.exit_position = None

    def play(self):
        self.make_walls()

        if self.render:
            pygame.init()
            pygame.display.set_caption('Exit')
            self.clock = pygame.time.Clock()
            self.display = pygame.display.set_mode(self.dimensions)

        self.agent_position = Coordinate(((self.dimensions[0] // self.block_size) // 2) * self.block_size, ((self.dimensions[1] // self.block_size) // 2) * self.block_size)
        # self.exit_position = self.generate_food()
        self.exit_position = Coordinate(self.block_size, self.block_size)

        while not self.at_exit:
            self.handle_events()
            self.agent.act(self)
            self.at_exit = self.agent_position.x == self.exit_position.x and self.agent_position.y == self.exit_position.y
            # self.update()
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
                    self.at_exit = True

    def act(self):
        vector = self.move_direction.to_vector()

        new_pos = Coordinate(self.agent_position.x + vector[0] * self.block_size, self.agent_position.y + vector[1] * self.block_size)

        # temp check if head is out of bounds
        if new_pos.x < self.block_size or new_pos.x >= self.dimensions[0]-self.block_size or new_pos.y < self.block_size or new_pos.y >= self.dimensions[1]-self.block_size:
            return

        self.agent_position = new_pos

    def draw(self):
        self.display.fill((0, 0, 0))
        self.draw_walls()
        self.draw_exit()
        self.draw_agent()
        pygame.display.update()

    def draw_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.display, (50, 50, 50), (wall.x, wall.y, self.block_size, self.block_size))

    def draw_agent(self):
        pygame.draw.rect(self.display, (255, 0, 0), (self.agent_position.x, self.agent_position.y, self.block_size, self.block_size))

    def draw_exit(self):
        pygame.draw.rect(self.display, (0, 255, 0), (self.exit_position.x, self.exit_position.y, self.block_size, self.block_size))


if __name__ == "__main__":
    agent = KeyboardAgent()
    egame = ExitGame(agent, True)
    egame.play()
