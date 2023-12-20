import datetime
import itertools
import random
from typing import List, Tuple
from enum import Enum
import math

import numpy as np

from snake_game import AbstractSnakeGame, Coordinate
from misc import Direction, Vector
from dataclasses import dataclass
import torch, torch.nn as nn


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
    state_tensor: torch.Tensor
    action_tensor: torch.Tensor


@dataclass
class LearningType(Enum):
    QLearningOffPolicy = 0
    SARSAOnPolicy = 1
    FittedQLearning = 2


class DeepQLearningLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discount_factor, reward, q1, q0):
        # instead of getting q from the q_values, we get it from the model
        # todo this model is wrong, it predicts action not q - actually maybe these are the same?
        loss = reward + discount_factor * q1.argmax().item() - q0
        lsq_loss = loss ** 2
        return lsq_loss


class DeepQLearningSnakeGame(AbstractSnakeGame):
    learning_rate = 0.7
    discount_factor = 0.5
    epsilon = 0.1

    # define neural net (https://github.com/blakeMilner/DeepQLearning/blob/master/deepqlearn.lua)
    # states and actions that go into the neural net (state0,action0),(state1,action1), ... , (stateN)
    # this variable controls the size of the temporal window

    # Number of past state/action pairs input to the network. 0 = agent lives in-the-moment :)
    sqs = [''.join(s) for s in list(itertools.product(*[['0', '1', '2']] * 4))]
    widths = ['0', '1', '4']
    heights = ['2', '3', '4']

    num_states = len(sqs) * len(widths) * len(heights)

    num_states = 6  # 4 for surroundings, 2 for food position, * 2 for temporal window
    num_actions = 4
    temporal_window = 2
    # net_inputs = (num_states + num_actions) * temporal_window + num_states
    net_inputs = num_states * temporal_window
    hidden_nodes = 16

    model = nn.Sequential(
        nn.Linear(net_inputs, hidden_nodes),
        nn.ReLU(),
        nn.Linear(hidden_nodes, hidden_nodes),
        nn.ReLU(),
        nn.Linear(hidden_nodes, num_actions)
    )

    # He initialization of weights
    def weights_init(layer_in):
        if isinstance(layer_in, nn.Linear):
            nn.init.kaiming_uniform(layer_in.weight)
            layer_in.bias.data.fill_(0.0)

    model.apply(weights_init)

    # set loss function
    criterion = DeepQLearningLoss()

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def __init__(self, state_dict=None, use_renderer=True):
        super().__init__(use_renderer)
        self.frametime = 50

        self.history: List[History] = []

        self.learning_type = LearningType.QLearningOffPolicy



    def get_action(self) -> Direction:
        state = self.get_state()

        # epsilon-greedy todo how does this work with model inference?
        # if random.random() < self.epsilon:
        #     action = random.randint(0, 3)
        #     action_direction = Direction(action)
        #     self.history.append(History(state, action_direction, None, None))
        #     return action_direction

        # NN forward pass
        # state to tensor
        state_tensor = torch.tensor([float(state.relative_food_direction[0]), float(state.relative_food_direction[1]),
                                     *[float(x) for x in list(state.surroundings)]])

        prev_state = self.history[-1].state if len(self.history) > 0 else None
        if prev_state is not None:
            state_tensor_prev = torch.tensor([float(prev_state.relative_food_direction[0]), float(prev_state.relative_food_direction[1]), *[float(x) for x in list(prev_state.surroundings)]])

        else:
            state_tensor_prev = torch.zeros_like(state_tensor)

        # concat prev state and current state
        state_tensor_all = torch.cat((state_tensor_prev, state_tensor))
        self.action_tensor = self.model(state_tensor_all)
        action = self.action_tensor.argmax().item()

        # take index of max value
        action_direction = Direction(action)
        self.history.append(History(state, action_direction, state_tensor_all, self.action_tensor))

        return action_direction

    def get_state(self) -> State:
        snake_head = self.snake[0]
        distance_to_food = self.food_location - snake_head

        if distance_to_food.x > 0:
            pos_x = '1'  # Food is to the right of the snake
        elif distance_to_food.x < 0:
            pos_x = '0'  # Food is to the left of the snake
        else:
            pos_x = '4'  # Food and snake are on the same X file

        if distance_to_food.y > 0:
            pos_y = '3'  # Food is below snake
        elif distance_to_food.y < 0:
            pos_y = '2'  # Food is above snake
        else:
            pos_y = '4'  # Food and snake are on the same Y file

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
            # todo fix

            # self.q_values[s0q][a0.value] += self.learning_rate * (reward - self.q_values[s0q][a0.value])


        # this updates the previous state's q-value based on the current state

        h1 = self.history[-1]  # current state
        h0 = self.history[-2]  # previous state

        # reward is defined as acquired AFTER the action is taken (SARSA), so this is previous-state reward
        if h1.state.food_position != h0.state.food_position:  # Snake ate a food, positive reward
            reward = 3
        elif h1.state.distance_to_food.length() < h0.state.distance_to_food.length():  # Snake is closer to the food, positive reward
            reward = 1
        # if snake next to tail, negative reward
        else:
            reward = -1  # Snake is further from the food, negative reward

        # discourage going back to where you came from to avoid oscillation
        # todo readd
        if len(self.history) > 2:
            sm1 = self.history[-3].state  # state before previous state
            if sm1 == h1.state:
                reward += -2

        # if s0.surroundings[0] == '2' or s0.surroundings[1] == '2' or s0.surroundings[2] == '2' or s0.surroundings[3] == '2':
        #     reward += -1

        # if any([s != '0' for s in s0.surroundings]):
        #     reward += -1
        q1 = self.model(h1.state_tensor)
        q0 = self.model(h0.state_tensor)[h0.action.value]


        # model backwards pass based on the reward
        self.optimizer.zero_grad()

        if self.snake_alive:
            loss = self.criterion(self.discount_factor, reward, q1, q0)
        else:
            loss = self.criterion(0, -5, q1, q0)
        loss.backward()
        self.optimizer.step()
        a = 0

if __name__ == "__main__":
    game_count = 0
    game_count_cap = 5000
    epsilon_trigger = 100
    desc = f"fromzero_e{epsilon_trigger}"
    filename = f"qlearning_{desc}_{int(datetime.datetime.now().timestamp())}_{game_count_cap}.txt"
    state_dict = None
    with open(filename, 'w') as f:
        f.write(f"Game,Score\n")
    while game_count < game_count_cap:
        game = DeepQLearningSnakeGame(state_dict, False)
        # if game_count % 100 == 0 and game_count > 90:
        #     game.epsilon *= 0.5
        if game_count > epsilon_trigger:
            game.epsilon = 0
        game.frametime = 50_000
        game.block_size = 10
        game.play()
        game_count += 1
        print(f"Games: {game_count}, Score: {game.get_score()}")
        state_dict = game.model.state_dict()

        with open(filename, 'a') as f:
            f.write(f"{game_count},{game.get_score()},{game.death_reason}\n")