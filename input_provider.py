import abc
import pygame
from enum import Enum


class Direction(Enum):
    NONE = -1
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def to_vector(self):
        match self:
            case Direction.UP:
                return 0, -1
            case Direction.DOWN:
                return 0, 1
            case Direction.LEFT:
                return -1, 0
            case Direction.RIGHT:
                return 1, 0
            case Direction.NONE:
                return 0, 0
            case other:
                raise ValueError(f"Invalid direction: {other}")


class InputProvider(metaclass=abc.ABCMeta):
    """Abstract base class for input providers."""

    @abc.abstractmethod
    def get_input(self) -> Direction:
        """Retrieve input from the user."""
        return NotImplemented


class KeyboardInputProvider(InputProvider):
    """Input provider for keyboard input."""

    def get_input(self):
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