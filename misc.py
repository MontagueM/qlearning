from dataclasses import dataclass
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


@dataclass
class Coordinate:
    x: int
    y: int

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


@dataclass
class Vector(Coordinate):
    def length(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
