import datetime
import itertools
import os
import random
from typing import List, Tuple
from enum import Enum
from collections import deque
from copy import deepcopy

import pygame
import numpy as np
from memory_profiler import profile

from keyboard import KeyboardSnakeGame
from snake_game import AbstractSnakeGame, Coordinate, DeathReason
from misc import Direction, Vector
from dataclasses import dataclass
import torch, torch.nn as nn

import matplotlib.pyplot as plt


plt.ion()


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
    grid: torch.Tensor

@dataclass
class Experience:
    state: State
    action: Direction
    reward: float
    next_state: State
    final_step: bool


@dataclass
class History:
    state: State
    action: Direction


@dataclass
class LearningType(Enum):
    QLearningOffPolicy = 0
    SARSAOnPolicy = 1
    FittedQLearning = 2


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

global_count = 0
losses = []
rewards = []


class DeepQLearningSnakeGame(AbstractSnakeGame):
    num_actions = 4
    hidden_nodes_1 = 256
    hidden_nodes_2 = hidden_nodes_1

    # mini-batches are preferable to larger 1. speed 2. performs better apparently
    # also more commonly referenced in papers like https://arxiv.org/pdf/1312.5602.pdf
    batch_size = 32
    train_start = batch_size*8
    mem_size = 100_000

    learning_rate = 0.001
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_min = 0.0
    eps_steps = 100_000

    # replay_memory = deque(maxlen=mem_size)
    replay_memory = {}

    target_model_update = 10_000

    num_frames = 4
    num_channels = 3

    kernel_1 = 3
    kernel_2 = 3
    stride_1 = 1
    stride_2 = 1
    padding = 0
    out_channels_1 = 16
    out_channels_2 = 32
    input_width = 10
    input_height = 10
    max_pool_kernel = 1
    conv1_output_width = (input_width - kernel_2 + 2 * padding) // stride_1 + 1
    conv1_output_height = (input_height - kernel_1 + 2 * padding) // stride_1 + 1
    max_pool_stride = 1
    dilation = 1
    maxpool1_output_width = (conv1_output_width + 2 * padding - dilation * (max_pool_kernel - 1) - 1) // max_pool_stride + 1
    maxpool1_output_height = (conv1_output_height + 2 * padding - dilation * (max_pool_kernel - 1) - 1) // max_pool_stride + 1
    conv2_output_width = (maxpool1_output_width - kernel_2 + 2 * padding) // stride_2 + 1
    conv2_output_height = (maxpool1_output_height - kernel_2 + 2 * padding) // stride_2 + 1
    maxpool2_output_width = (conv2_output_width + 2 * padding - dilation * (max_pool_kernel - 1) - 1) // max_pool_stride + 1
    maxpool2_output_height = (conv2_output_height + 2 * padding - dilation * (max_pool_kernel - 1) - 1) // max_pool_stride + 1
    # linear_input_size = maxpool2_output_width * maxpool2_output_height * out_channels_2
    linear_input_size = 1152
    print(f"linear input size: {linear_input_size}")

    features = 256
    model = nn.Sequential(
        nn.Conv2d(num_channels, out_channels_1, kernel_size=kernel_1, stride=stride_1, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels_1, out_channels_2, kernel_size=kernel_2, stride=stride_2, padding=padding),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d(6),
        nn.Flatten(),
        nn.Linear(32*6*6, features),
        nn.ReLU(),
        nn.Linear(features, num_actions)
    )
    # target_model.compile()

    target_model = deepcopy(model)
    # model.compile()

    # He initialization of weights

    target_model.apply(weights_init)
    model.apply(weights_init)

    # set loss function
    # criterion = DeepQLearningLoss()
    criterion = nn.MSELoss()

    # set optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    def __init__(self, state_dict=None, use_renderer=True):
        super().__init__(use_renderer)
        self.frametime = 50

        self.history: List[History] = []

        self.learning_type = LearningType.QLearningOffPolicy

        self.local_count = 0
        self.action_every_n_frames = 1

        self.food_iterations = [0]

        self.do_train = True

        self.grid_since_last_food = []

    def play(self):
        self.grid_since_last_food = np.zeros((self.dimensions[0] // self.block_size, self.dimensions[1] // self.block_size))
        self.loop_count = 0
        super().play()

    def get_action(self) -> Direction:
        if self.local_count % self.action_every_n_frames != 0:
            return Direction.NONE

        state = self.get_state()

        # epsilon-greedy todo how does this work with model inference?
        if random.random() < self.epsilon or global_count < self.train_start and self.do_train:
            action = random.randint(0, 3)
            action_direction = Direction(action)
            self.history.append(History(state, action_direction))
            return action_direction

        if global_count == self.train_start and self.do_train:
            print("Starting training")

        # NN forward pass
        # state to tensor
        state_tensor = state.grid

        # prev_state_tensor = self.history[-1].state.tensor()
        # if prev_state is not None:
        #
        # else:
        #     state_tensor_prev = torch.zeros_like(state_tensor)

        # concat prev state and current state
        # state_tensor_all = torch.cat((prev_state_tensor, state_tensor))
        state_tensor = state_tensor.unsqueeze(0)

        # on-policy if self.model, off-policy if self.test_model
        action_tensor = self.model(state_tensor)
        action = action_tensor.argmax().item()

        # take index of max value
        action_direction = Direction(action)
        self.history.append(History(state, action_direction))

        if len(self.history) > 1000:
            self.history.pop(0)

        return action_direction

    def get_state(self) -> State:
        # convert stateful info into 2d grid with 3 channels
        grid = np.zeros((self.dimensions[0] // self.block_size, self.dimensions[1] // self.block_size, 3))
        # food is green
        grid[self.food_location.x // self.block_size, self.food_location.y // self.block_size, 1] = 1
        # snake head is red
        grid[self.snake[0].x // self.block_size, self.snake[0].y // self.block_size, 0] = 1
        # snake body is white
        for body_part in self.snake[1:]:
            grid[body_part.x // self.block_size, body_part.y // self.block_size, :] = 1
        # add walls
        for wall in self.walls:
            grid[wall.x // self.block_size, wall.y // self.block_size, :] = 0.5

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

        # permute to fit BATCH x CHANNELS x HEIGHT x WIDTH
        grid = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1)
        return State(distance_to_food, (pos_x, pos_y), surroundings, self.food_location, grid)

    def update(self):
        super().update()

        self.update_qvalues()

        # if self.get_score() > 30:
        #     self.frametime = 20

    def update_qvalues(self):
        global global_count
        if len(self.history) < 2:
            self.local_count += 1
            global_count += 1
            return


        current = self.history[-1]  # current state
        previous = self.history[-2]  # previous state

        # reward is defined as acquired AFTER the action is taken (SARSA), so this is previous-state reward
        if current.state.food_position != previous.state.food_position:  # Snake ate a food, positive reward
            reward = 3
            self.food_iterations.append(self.local_count - self.food_iterations[-1])
            self.grid_since_last_food = np.zeros((self.dimensions[0] // self.block_size, self.dimensions[1] // self.block_size))
        elif abs(current.state.distance_to_food.x) < abs(previous.state.distance_to_food.x) or abs(current.state.distance_to_food.y) < abs(previous.state.distance_to_food.y):  # Snake is closer to the food, positive reward
            reward = 1
        # if snake next to tail, negative reward
        else:
            reward = -1  # Snake is further from the food, negative reward

        # snake can get caught in loop and never learn about the negative reward for that, so kill if we detect
        # maybe add noise to the Q-values on inference?
        # better idea is store the grid positions of the snake and if it ever repeats, give it a negative reward
        # this will alter the path it will take next, which will help to avoid loops

        detect_loops = True
        if detect_loops:
            snake_head = self.snake[0]
            if self.grid_since_last_food[snake_head.x // self.block_size, snake_head.y // self.block_size] == 1:
                # reward = -1
                self.loop_count += 1

                if not self.do_train and self.loop_count > 10:# and self.epsilon < 0.1:
                    self.snake_alive = False
                    self.death_reason = DeathReason.LOOP

            self.grid_since_last_food[snake_head.x // self.block_size, snake_head.y // self.block_size] = 1

        rewards.append(reward)

        # fix
        if self.dimensions[0] not in self.replay_memory:
            self.replay_memory[self.dimensions[0]] = deque(maxlen=self.mem_size)
        if not self.snake_alive:
            reward = -5
            self.replay_memory[self.dimensions[0]].append(
                Experience(current.state, current.action, float(reward), current.state, not self.snake_alive))
        else:
            self.replay_memory[self.dimensions[0]].append(Experience(previous.state, previous.action, float(reward), current.state, not self.snake_alive))

        if self.do_train:
            loss = self.train()
            losses.append(loss)

        self.local_count += 1
        global_count += 1

        if global_count % self.target_model_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("Updated target model")

        if global_count % 5_000 == 0:
            print(f"Global count: {global_count}")

    def train(self) -> float:
        if global_count < self.train_start:
            return 0

        if len(self.replay_memory[self.dimensions[0]]) < self.batch_size:
            batch = self.replay_memory[self.dimensions[0]]
        else:
            batch = random.sample(self.replay_memory[self.dimensions[0]], self.batch_size)
        rewards = torch.tensor([exp.reward for exp in batch])
        states = torch.stack([exp.state.grid for exp in batch])
        actions = torch.tensor([exp.action.value for exp in batch])
        # one-hot encode actions as all action values are equally valuable, e.g. 3 is not better than 2
        # actions_onehot = self.get_onehot(actions)
        next_states = torch.stack([exp.next_state.grid for exp in batch])
        final_steps = torch.tensor([exp.final_step for exp in batch])


        batch_indices = torch.arange(len(actions))

        discount_factor_tensor = torch.full_like(final_steps, self.discount_factor, dtype=torch.float32)
        discount_factor_tensor = torch.where(final_steps, torch.zeros_like(discount_factor_tensor), discount_factor_tensor)

        predicted = self.model(states)
        y = predicted.clone()

        # update each action with the expected q value, leave the rest the same as did not change
        # DDQN https://arxiv.org/pdf/1509.06461.pdf
        # use online model to pick action (y), use target model to evaluate q value (q)
        q = torch.max(self.target_model(next_states), dim=1).values
        y[batch_indices, actions] = rewards + discount_factor_tensor * q

        self.model.train()

        self.optimizer.zero_grad()
        loss = self.criterion(predicted, y)
        loss_float = loss.item()
        loss.backward()
        self.optimizer.step()

        self.model.eval()

        return loss_float

# todo reconsider how correct this is given the concept that rewards are given when LEAVING a state
@dataclass
class RecordHistory:
    state: np.ndarray
    snake: List[Coordinate]
    food_location: Coordinate
    action: Direction
    eaten_food: bool
    died: bool


class RecordSnakeGame(KeyboardSnakeGame):
    def __init__(self):
        super().__init__()

        self.history: List[RecordHistory] = []

    def update(self):
        super().update()

        if self.move_direction == Direction.NONE:
            return

        food_eaten = self.food_location != self.history[-1].food_location if self.history else False
        self.history.append(RecordHistory(deepcopy(self.get_state()), deepcopy(self.snake), deepcopy(self.food_location), self.get_action(), food_eaten, not self.snake_alive))

    def get_state(self) -> np.ndarray:
        # convert stateful info into 2d grid with 3 channels
        grid = np.zeros((self.dimensions[0] // self.block_size, self.dimensions[1] // self.block_size, 3))
        # food is green
        grid[self.food_location.x // self.block_size, self.food_location.y // self.block_size, 1] = 1
        # snake head is red
        grid[self.snake[0].x // self.block_size, self.snake[0].y // self.block_size, 0] = 1
        # snake body is white
        for body_part in self.snake[1:]:
            grid[body_part.x // self.block_size, body_part.y // self.block_size, :] = 1

        return grid


@dataclass
class Span:
    start: int
    end: int = lambda self: self.start + self.size
    size: int = lambda self: self.end - self.start


class PlaybackSnakeGame(AbstractSnakeGame):
    def __init__(self, history: List[RecordHistory]):
        super().__init__(use_renderer=True)

        self.human_history: List[RecordHistory] = history
        self.history: List[History] = []

        num_actions = 4
        hidden_nodes_1 = 256
        hidden_nodes_2 = hidden_nodes_1

        # mini-batches are preferable to larger 1. speed 2. performs better apparently
        # also more commonly referenced in papers like https://arxiv.org/pdf/1312.5602.pdf
        self.batch_size = 32
        train_start = self.batch_size * 8
        mem_size = 100_000

        learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.1
        epsilon_min = 0.0
        eps_steps = 100_000

        # replay_memory = deque(maxlen=mem_size)
        replay_memory = {}

        target_model_update = 10_000

        num_frames = 4
        num_channels = 3

        kernel_1 = 3
        kernel_2 = 3
        stride_1 = 1
        stride_2 = 1
        padding = 0
        out_channels_1 = 16
        out_channels_2 = 32
        input_width = 10
        input_height = 10
        max_pool_kernel = 1
        conv1_output_width = (input_width - kernel_2 + 2 * padding) // stride_1 + 1
        conv1_output_height = (input_height - kernel_1 + 2 * padding) // stride_1 + 1
        max_pool_stride = 1
        dilation = 1
        maxpool1_output_width = (conv1_output_width + 2 * padding - dilation * (
                    max_pool_kernel - 1) - 1) // max_pool_stride + 1
        maxpool1_output_height = (conv1_output_height + 2 * padding - dilation * (
                    max_pool_kernel - 1) - 1) // max_pool_stride + 1
        conv2_output_width = (maxpool1_output_width - kernel_2 + 2 * padding) // stride_2 + 1
        conv2_output_height = (maxpool1_output_height - kernel_2 + 2 * padding) // stride_2 + 1
        maxpool2_output_width = (conv2_output_width + 2 * padding - dilation * (
                    max_pool_kernel - 1) - 1) // max_pool_stride + 1
        maxpool2_output_height = (conv2_output_height + 2 * padding - dilation * (
                    max_pool_kernel - 1) - 1) // max_pool_stride + 1
        # linear_input_size = maxpool2_output_width * maxpool2_output_height * out_channels_2
        linear_input_size = 1152
        print(f"linear input size: {linear_input_size}")

        features = 256
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, out_channels_1, kernel_size=kernel_1, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size=kernel_2, stride=stride_2, padding=padding),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(6),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, features),
            nn.ReLU(),
            nn.Linear(features, num_actions)
        )

        self.model.apply(weights_init)

        # set loss function
        # criterion = DeepQLearningLoss()
        self.criterion = nn.MSELoss()

        # set optimizer
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

        self.replay_memory = deque(maxlen=mem_size)

    def play(self):
        self.make_walls()

        pygame.init()
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.dimensions)

        # split history up based on when food was eaten
        split_history = [[]]
        for h in self.human_history:
            split_history[-1].append(h)
            if h.eaten_food:
                split_history.append([])

        for h in split_history[:-1][::-1]:  # skip last one as it is death, todo consider doing this but as a negative reward?
            self.playback(h)
            quit(2)

    def get_action(self) -> Direction:
        state = self.get_state()

        if random.random() < self.epsilon:
            action = random.randint(0, 3)
            action_direction = Direction(action)
            self.history.append(History(state, action_direction))
            return action_direction

        state_tensor = state.grid
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = self.model(state_tensor)
        action = action_tensor.argmax().item()

        # take index of max value
        action_direction = Direction(action)
        self.history.append(History(state, action_direction))

        return action_direction

    def get_state(self) -> State:
        # convert stateful info into 2d grid with 3 channels
        grid = np.zeros((self.dimensions[0] // self.block_size, self.dimensions[1] // self.block_size, 3))
        # food is green
        grid[self.food_location.x // self.block_size, self.food_location.y // self.block_size, 1] = 1
        # snake head is red
        grid[self.snake[0].x // self.block_size, self.snake[0].y // self.block_size, 0] = 1
        # snake body is white
        for body_part in self.snake[1:]:
            grid[body_part.x // self.block_size, body_part.y // self.block_size, :] = 1
        # add walls
        for wall in self.walls:
            grid[wall.x // self.block_size, wall.y // self.block_size, :] = 0.5

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

        # permute to fit BATCH x CHANNELS x HEIGHT x WIDTH
        grid = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1)
        return State(distance_to_food, (pos_x, pos_y), surroundings, self.food_location, grid)

    def playback(self, history: List[RecordHistory]):
        """
        history is a list of recorded human game history,
        starting when reward was spawned and ending when reward was acquired.
        this func iterates over this history backwards, letting the model learn.
        """

        current_back_index = len(history) - 2
        max_repeats = 5000

        while current_back_index >= 0:
            h = history[current_back_index]

            reach_reward_success: int = 0
            repeats: int = 0
            success_ratio: float = 0.75
            repeat_min = 10


            epsilon_start = 1.0
            epsilon_min = 0.01
            epsilon_len = 50

            repeat_min = epsilon_len

            has_success = lambda: repeats > repeat_min and reach_reward_success / repeats > success_ratio


            while repeats < max_repeats and not has_success():
                self.epsilon = max(epsilon_min, epsilon_start * (1 - repeats / epsilon_len))
                print(f"Repeats: {repeats}, Success ratio: {reach_reward_success / repeats if repeats > 0 else 0}, Epsilon: {self.epsilon}")
                self.snake_alive = True
                its = 0
                self.snake = deepcopy(h.snake)
                self.food_location = h.food_location


                self.draw()
                self.clock.tick(self.frametime)

                while self.snake_alive:
                    self.handle_events()
                    self.check_collision()
                    # todo consider case of rewarding any of the same position instead
                    # of just the action similarity
                    human_action = None
                    if current_back_index + its < len(history):
                        human_action = history[current_back_index + its].action
                        # print(f"HUMAN ACTION: {human_action} vs MACHINE ACTION: {self.history[-1].action}")
                    self.update_model(human_action)

                    self.draw()
                    self.clock.tick(self.frametime)

                    if self.food_eaten():
                        reach_reward_success += 1
                        break

                    its += 1

                repeats += 1

            current_back_index -= 1


            if repeats >= max_repeats:
                print("Max repeats reached")
                break

            if has_success():
                print("Success")
                pass

        pass

    def update_model(self, human_action):
        if len(self.history) < 2:
            return

        current = self.history[-1]
        previous = self.history[-2]

        # todo consider impact of not using prev != current logic
        if self.food_eaten():
            reward = 10
        elif abs(current.state.distance_to_food.x) < abs(previous.state.distance_to_food.x) or abs(current.state.distance_to_food.y) < abs(previous.state.distance_to_food.y):
            reward = 1
        else:
            reward = -1

        # if the move is identical to the human, reward stronger
        if human_action == current.action:
            reward += 4

        if not self.snake_alive:
            reward = -5

        if not self.snake_alive or self.food_eaten():
            self.replay_memory.append(
                Experience(current.state, current.action, float(reward), current.state, True))
        else:
            self.replay_memory.append(Experience(previous.state, previous.action, float(reward), current.state, False))


        loss = self.train()
        print(f"Loss: {loss}")

    def train(self) -> float:
        if len(self.replay_memory) < self.batch_size:
            batch = self.replay_memory
        else:
            batch = random.sample(self.replay_memory, self.batch_size)
        rewards = torch.tensor([exp.reward for exp in batch])
        states = torch.stack([exp.state.grid for exp in batch])
        actions = torch.tensor([exp.action.value for exp in batch])
        # one-hot encode actions as all action values are equally valuable, e.g. 3 is not better than 2
        # actions_onehot = self.get_onehot(actions)
        next_states = torch.stack([exp.next_state.grid for exp in batch])
        final_steps = torch.tensor([exp.final_step for exp in batch])


        batch_indices = torch.arange(len(actions))

        discount_factor_tensor = torch.full_like(final_steps, self.discount_factor, dtype=torch.float32)
        discount_factor_tensor = torch.where(final_steps, torch.zeros_like(discount_factor_tensor), discount_factor_tensor)

        predicted = self.model(states)
        y = predicted.clone()

        # update each action with the expected q value, leave the rest the same as did not change
        # DDQN https://arxiv.org/pdf/1509.06461.pdf
        # use online model to pick action (y), use target model to evaluate q value (q)
        q = torch.max(self.model(next_states), dim=1).values
        y[batch_indices, actions] = rewards + discount_factor_tensor * q

        self.model.train()

        self.optimizer.zero_grad()
        loss = self.criterion(predicted, y)
        loss_float = loss.item()
        loss.backward()
        self.optimizer.step()

        self.model.eval()

        return loss_float



if __name__ == "__main__":
    game_count = 0
    game_count_cap = 300
    epsilon_trigger = 0
    desc = f"fromzero_e{epsilon_trigger}"
    filename = f"qlearning_{desc}_{int(datetime.datetime.now().timestamp())}_{game_count_cap}.txt"
    state_dict = None
    scores = []
    _losses = []
    _rewards_sum = []
    _rewards_mean = []
    tail_deaths = []
    wall_deaths = []
    loop_deaths = []
    death_reasons = []
    iterations = []
    iteration_foods = []
    running_trend = 50
    epsilons = []

    # mps seems to be better at large batch sizes e.g. 1024-2048
    # torch.set_default_device("mps")
    # device = torch.device("mps")
    torch.set_default_device("cpu")
    device = torch.device("cpu")

    with open(filename, 'w') as f:
        f.write(f"Game,Score\n")

    timestamp = int(datetime.datetime.now().timestamp())
    use_checkpoint = False
    checkpoint_file = "data/cnn/1703621993/model_1000.pth"
    action_every_n_frames = 1

    record = False
    if record:
        game = RecordSnakeGame()
        game.dimensions = (150, 150)
        game.frametime = 5
        game.play()

        # save the game
        data = {}
        data["dimensions"] = game.dimensions
        data["num_iterations"] = len(game.history)
        data["score"] = game.get_score()
        data["history"] = game.history

        directory = f"data/recorded/{timestamp}"
        os.makedirs(directory, exist_ok=True)
        torch.save(data, f"{directory}/game_{len(game.history)}_{game.get_score()}.pth")
        quit(0)

    use_recorded = True
    recorded_path = "data/recorded/1703748853/game_28_3.pth"
    if use_recorded:
        data = torch.load(recorded_path)
        game = PlaybackSnakeGame(data["history"])
        game.dimensions = data["dimensions"]
        game.frametime = 50_000
        game.play()
        quit(0)
