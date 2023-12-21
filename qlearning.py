import datetime
import itertools
import random
from typing import List, Tuple
from enum import Enum

import numpy as np

from snake_game import AbstractSnakeGame, Coordinate
from misc import Direction, Vector
from dataclasses import dataclass


@dataclass
class QState:
    relative_food_direction: tuple
    surroundings: str

    def __hash__(self):
        return hash((self.relative_food_direction, self.surroundings))


@dataclass
class State:
    distance_to_food: Vector
    relative_food_direction: tuple
    surroundings: str
    food_position: Coordinate  # only used to know if snake ate food

    def q_state(self):
        return QState(self.relative_food_direction, self.surroundings)


@dataclass
class History:
    state: State
    action: Direction


@dataclass
class LearningType(Enum):
    QLearningOffPolicy = 0
    SARSAOnPolicy = 1
    FittedQLearning = 2


class QLearningSnakeGame(AbstractSnakeGame):
    learning_rate = 0.7
    discount_factor = 0.5
    epsilon = 0.1

    def __init__(self, q_values, use_renderer=True):
        super().__init__(use_renderer)
        self.frametime = 50

        self.history: List[History] = []
        self.q_values = q_values

        self.learning_type = LearningType.QLearningOffPolicy

    def get_action(self) -> Direction:
        state = self.get_state()

        # epsilon-greedy
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            state_scores = self.q_values.get(state.q_state())
            action = state_scores.index(max(state_scores))

        action_direction = Direction(action)
        self.history.append(History(state, action_direction))

        return action_direction

    def get_state(self) -> State:
        snake_head = self.snake[0]
        distance_to_food = self.food_location - snake_head

        if distance_to_food.x > 0:
            pos_x = '1'  # Food is to the right of the snake
        elif distance_to_food.x < 0:
            pos_x = '0'  # Food is to the left of the snake
        else:
            pos_x = 'NA'  # Food and snake are on the same X file

        if distance_to_food.y > 0:
            pos_y = '3'  # Food is below snake
        elif distance_to_food.y < 0:
            pos_y = '2'  # Food is above snake
        else:
            pos_y = 'NA'  # Food and snake are on the same Y file

        sqs = [
            Coordinate(snake_head.x - self.block_size, snake_head.y),
            Coordinate(snake_head.x + self.block_size, snake_head.y),
            Coordinate(snake_head.x, snake_head.y - self.block_size),
            Coordinate(snake_head.x, snake_head.y + self.block_size),
        ]

        surrounding_list = []
        for sq in sqs:
            if sq.x < 0 or sq.y < 0:  # off-screen left or top
                surrounding_list.append('1')
            elif sq.x >= self.dimensions[0] or sq.y >= self.dimensions[1]:  # off-screen right or bottom
                surrounding_list.append('1')
            elif sq in self.snake[1:-1]:  # part of tail
                surrounding_list.append('2')
            else:
                surrounding_list.append('0')
        surroundings = ''.join(surrounding_list)

        return State(distance_to_food, (pos_x, pos_y), surroundings, self.food_location)

    def update(self):
        super().update()

        self.update_qvalues()

        # if self.get_score() > 30:
        #     self.frametime = 20

    def update_qvalues(self):
        if len(self.history) < 2:
            return

        # we have to special case death as there is no next state to register a penalty
        if not self.snake_alive:
            s0 = self.history[-1].state
            a0 = self.history[-1].action
            s0q = s0.q_state()
            reward = -5
            self.q_values[s0q][a0.value] += self.learning_rate * (reward - self.q_values[s0q][a0.value])


        # this updates the previous state's q-value based on the current state

        s1 = self.history[-1].state  # current state
        s0 = self.history[-2].state  # previous state
        a0 = self.history[-2].action  # action taken at previous state

        # reward is defined as acquired AFTER the action is taken (SARSA), so this is previous-state reward
        if s1.food_position != s0.food_position:  # Snake ate a food, positive reward
            reward = 3
        elif abs(s1.distance_to_food.x) < abs(s0.distance_to_food.x) or abs(s1.distance_to_food.y) < abs(s0.distance_to_food.y):  # Snake is closer to the food, positive reward
            reward = 1
        else:
            reward = -1  # Snake is further from the food, negative reward

        # discourage going back to where you came from to avoid oscillation
        if len(self.history) > 2:
            sm1 = self.history[-3].state  # state before previous state
            if sm1 == s1:
                reward += -1

        # if s0.surroundings[0] == '2' or s0.surroundings[1] == '2' or s0.surroundings[2] == '2' or s0.surroundings[3] == '2':
        #     reward += -1

        # if any([s != '0' for s in s0.surroundings]):
        #     reward += -1

        s1q = s1.q_state()
        s0q = s0.q_state()

        match self.learning_type:
            case LearningType.SARSAOnPolicy:
                a1 = self.history[-1].action
                self.q_values[s0q][a0.value] += self.learning_rate * (reward + self.discount_factor * self.q_values[s1q][a1.value] - self.q_values[s0q][a0.value])
            case LearningType.QLearningOffPolicy:
                self.q_values[s0q][a0.value] += self.learning_rate * (reward + self.discount_factor * max(self.q_values[s1q]) - self.q_values[s0q][a0.value])
            # case LearningType.FittedQLearning:
            #     loss = reward + self.discount_factor * max(self.q_values[s1q]) - self.q_values[s0q][a0.value]
            #     lsq_loss = loss ** 2
            #     phi = None
            #     phi_update = self.learning_rate * loss * delta_phi
            #     phi += phi_update
            case other:
                raise ValueError(f"Invalid learning type: {other}")


if __name__ == "__main__":
    use_cached_q_values = False

    if use_cached_q_values:
        q_values = np.load('q_values.npy', allow_pickle=True).item()
    else:
        sqs = [''.join(s) for s in list(itertools.product(*[['0', '1', '2']] * 4))]
        widths = ['0', '1', 'NA']
        heights = ['2', '3', 'NA']

        states = {}
        for i in widths:
            for j in heights:
                for k in sqs:
                    states[QState((i, j), k)] = [random.uniform(-1, 1) for _ in range(4)]
        q_values = states

    game_count = 0
    game_count_cap = 5000
    dump_every_n_games = 100
    epsilon_trigger = 100
    desc = f"fromzero_e{epsilon_trigger}"
    filename = f"qlearning_{desc}_{int(datetime.datetime.now().timestamp())}_{game_count_cap}.txt"
    with open(filename, 'w') as f:
        f.write(f"Game,Score\n")
    while game_count < game_count_cap:
        game = QLearningSnakeGame(q_values, False)
        # if game_count % 100 == 0 and game_count > 90:
        #     game.epsilon *= 0.5
        if game_count > epsilon_trigger:
            game.epsilon = 0
        game.frametime = 50_000
        game.block_size = 10
        game.play()
        game_count += 1
        print(f"Games: {game_count}, Score: {game.get_score()}")
        q_values = game.q_values

        if game_count % dump_every_n_games == 0:
            np.save(f"q_values.npy", q_values)

        with open(filename, 'a') as f:
            f.write(f"{game_count},{game.get_score()},{game.death_reason}\n")