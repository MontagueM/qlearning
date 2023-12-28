import datetime
import itertools
import os
import random
from typing import List, Tuple
from enum import Enum
from collections import deque
from copy import deepcopy

import numpy as np
from memory_profiler import profile

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
    dead: bool


@dataclass
class History:
    state: State
    action: Direction
    log_action_probabilities: torch.Tensor
    reward: float


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


class ReinforcePolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_model, discount_factor, rewards, states, actions, next_states, dead):
        # just MSE loss of expected q values vs actual q values, but without that explicit calculation
        q0_actions = target_model(states)
        batch_indices = torch.arange(len(actions))
        q0 = q0_actions[batch_indices, actions]
        q1 = torch.max(target_model(next_states), dim=1).values
        # discount factor is 0 if dead
        discount_factor_tensor = torch.full_like(dead, discount_factor, dtype=torch.float32)
        discount_factor_tensor = torch.where(dead, torch.zeros_like(discount_factor_tensor), discount_factor_tensor)
        loss = rewards + discount_factor_tensor * q1 - q0
        lsq_loss = loss ** 2

        if any(dead):
            # print("dead")
            pass

        return lsq_loss.mean()


class REINFORCESnakeGame(AbstractSnakeGame):
    num_actions = 4
    hidden_nodes_1 = 256
    hidden_nodes_2 = hidden_nodes_1

    # mini-batches are preferable to larger 1. speed 2. performs better apparently
    # also more commonly referenced in papers like https://arxiv.org/pdf/1312.5602.pdf
    batch_size = 32
    train_start = batch_size*8
    mem_size = 100_000

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
        nn.Linear(features, num_actions),
        nn.Softmax(dim=1)
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

    learning_rate = 2**-13
    learning_rate = 0.001
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        self.local_count += 1

    def get_action(self) -> Direction:
        if self.local_count % self.action_every_n_frames != 0:
            return Direction.NONE

        state = self.get_state()

        if global_count == self.train_start and self.do_train:
            print("Starting training")

        # NN forward pass
        state_tensor = state.grid
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = self.model(state_tensor)
        log_action_probabilities = torch.log(action_tensor)
        # choose based on probability of softmax
        action = torch.multinomial(action_tensor, 1)
        action = action.item()

        # take index of max value
        action_direction = Direction(action)

        self.history.append(History(state, action_direction, log_action_probabilities, 0))

        return action_direction

    def get_state(self) -> State:
        # convert stateful info into 2d grid with 1 channel, its 3 but combine for efficiency
        grid = np.zeros((self.dimensions[0] // self.block_size, self.dimensions[1] // self.block_size, 3))
        # food is green (0.5)
        grid[self.food_location.x // self.block_size, self.food_location.y // self.block_size, 1] = 1
        # snake head is red (0.25)
        grid[self.snake[0].x // self.block_size, self.snake[0].y // self.block_size, 0] = 1
        # snake body is white (1)
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

        if len(self.history) < 2:
            return

        # reward
        current = self.history[-1]  # current state
        previous = self.history[-2]  # previous state

        if self.food_eaten():
            reward = 0.3
        elif not self.snake_alive:
            reward = -1
        elif abs(current.state.distance_to_food.x) < abs(previous.state.distance_to_food.x) or abs(current.state.distance_to_food.y) < abs(previous.state.distance_to_food.y):
            reward = 0.1
        else:
            reward = -0.1

        self.history[-1].reward = reward


def gradients_wrt_params(
    net: torch.nn.Module, loss_tensor: torch.Tensor
):
    # Dictionary to store gradients for each parameter
    # Compute gradients with respect to each parameter
    for name, param in net.named_parameters():
        g = torch.autograd.grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = g


def update_params(net: torch.nn.Module, lr: float) -> None:
    # Update parameters for the network
    for name, param in net.named_parameters():
        param.data += lr * param.grad


if __name__ == "__main__":
    game_count = 0
    game_count_cap = 20_000
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
    checkpoint_file = "data/cnn/1703695105/model_1700.pth"
    action_every_n_frames = 1

    histories = []

    while game_count < game_count_cap:
        game = REINFORCESnakeGame(state_dict, True)

        # randomise dimensions between 50-250, for now always square
        dim = random.randint(5, 10) * 10 * 2
        game.dimensions = (dim, dim)
        # print(f"Dimensions: {game.dimensions}")
        game.dimensions = (110, 110)
        if game_count == 0:
            num_params = sum(p.numel() for p in game.model.parameters())
            print(f"Number of parameters: {num_params}, mean weight: {sum(p.sum() for p in game.model.parameters()) / num_params}")

            if use_checkpoint:
                checkpoint = torch.load(checkpoint_file, map_location=device)
                game.model.load_state_dict(checkpoint["model_state_dict"])
                game.target_model.load_state_dict(checkpoint["target_model_state_dict"])
                game.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                print(f"New mean weight: {sum(p.sum() for p in game.model.parameters()) / num_params}")

            game.model.to(device)
            game.target_model.to(device)

            game.model.eval()
            game.target_model.eval()

        if use_checkpoint:
            # todo figure out why setting this to False causes the game to stop working
            # my guess is that something in the model is not initialized properly
            game.do_train = False

        epsilons.append(game.epsilon)
        game.frametime = 50_000
        if use_checkpoint:
            game.frametime = 50
        game.action_every_n_frames = action_every_n_frames
        rewards = []
        losses = []
        game.play()
        histories.append(game.history)

        # train
        discount_factor = 0.99
        total_loss = 0
        total_reward = 0
        for t, h in enumerate(game.history):
            empirical_discounted_reward = 0
            for k in range(t+1, len(game.history)):
                empirical_discounted_reward += discount_factor ** k * game.history[k].reward

            total_reward += empirical_discounted_reward
            game.optimizer.zero_grad()
            policy_loss = -h.log_action_probabilities[0][h.action.value]
            total_loss += policy_loss.item()
            gradients_wrt_params(game.model, policy_loss)
            update_params(game.model, game.learning_rate * discount_factor ** t * empirical_discounted_reward)

        losses.append(total_loss)
        rewards.append(total_reward)
        print(f"Loss: {total_loss}")
        # if not losses:
        #     losses.append(0)
        game_count += 1
        print(f"Games: {game_count}, Score: {game.get_score()}, Epsilon: {game.epsilon}, Loss: {losses[-1]}")
        state_dict = game.model.state_dict()

        with open(filename, 'a') as f:
            f.write(f"{game_count},{game.get_score()},{game.death_reason}\n")

        scores.append(game.get_score())
        _losses.append(np.mean(losses))
        _rewards_sum.append(np.sum(rewards))
        _rewards_mean.append(np.mean(rewards))
        death_reasons.append(game.death_reason)
        iterations.append(game.local_count / game.get_score() if game.get_score() != 0 else game.local_count)
        iteration_foods.append(np.mean(game.food_iterations))

        if game_count % running_trend == 0:
            # proportion of deaths
            tail_deaths.append(death_reasons.count(DeathReason.TAIL) / len(death_reasons))
            wall_deaths.append(death_reasons.count(DeathReason.WALL) / len(death_reasons))
            loop_deaths.append(death_reasons.count(DeathReason.LOOP) / len(death_reasons))
            death_reasons = []

        num_almost_zero_weights = 0
        threshold = 1e-3
        for name, param in game.model.named_parameters():
            num_almost_zero_weights += torch.sum(torch.abs(param) < threshold).item()

        sparsity = num_almost_zero_weights / sum(p.numel() for p in game.model.parameters())
        # print(f"Sparsity: {sparsity}")


        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.close('all')

        plt.clf()

        fig, axs = plt.subplots(7, figsize=(5, 10), sharex=True)
        fig.subplots_adjust(top=0.95)
        if use_checkpoint:
            fig.suptitle(f"Inference on {checkpoint_file}")
        else:
            fig.suptitle(f'Train {timestamp}: LR={game.learning_rate}, EPS_MIN={game.epsilon_min}, DIMS={game.dimensions[0]}x{game.dimensions[1]}')

        running_trend_x = [(1+i) * running_trend for i in range(len(tail_deaths))]

        axs[0].set_ylabel('Score')
        axs[0].plot(scores, 'k')
        # running_trend on score
        scores_mean = [np.mean(scores[i*running_trend:i*running_trend + running_trend]) for i in range(len(tail_deaths))]
        axs[0].plot(running_trend_x, scores_mean, 'r')
        # 1 std dev
        axs[0].fill_between(running_trend_x, [scores_mean[i] - np.std(scores[i*running_trend:i*running_trend + running_trend]) for i in range(len(tail_deaths))], [scores_mean[i] + np.std(scores[i*running_trend:i*running_trend + running_trend]) for i in range(len(tail_deaths))], alpha=0.3, color="red")
        axs[0].set_ylim(ymin=0)
        axs[0].text(len(scores) - 1, scores[-1], str(scores[-1]))

        axs[1].set_ylabel('Loss')
        axs[1].plot(_losses, 'k')
        # axs[1].set_ylim(ymin=0)
        axs[1].text(len(losses) - 1, losses[-1], str(losses[-1]))

        # axs[2].set_ylabel('Rew.S')
        # axs[2].plot(_rewards_sum, 'k')
        # # running_trend on reward sum
        # rewards_sum_mean = [np.mean(_rewards_sum[i*running_trend:i*running_trend + running_trend]) for i in range(len(tail_deaths))]
        # axs[2].plot(running_trend_x, rewards_sum_mean, 'r')
        # axs[2].set_ylim(ymin=0)
        # axs[2].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))

        axs[2].set_ylabel('Rew.M')
        axs[2].plot(_rewards_mean, 'k')
        # axs[2].set_ylim(ymin=0)
        axs[2].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))

        axs[3].set_ylabel('Death')
        axs[3].plot(running_trend_x, tail_deaths, 'x', label="Tail", color="red")
        axs[3].plot(running_trend_x, wall_deaths, 'x', label="Wall", color="blue")
        axs[3].plot(running_trend_x, loop_deaths, 'x', label="Loop", color="green")
        axs[3].set_ylim(ymin=0, ymax=1)
        axs[3].legend(loc="upper left")

        axs[4].set_ylabel('Eps')
        axs[4].plot(epsilons, 'k')
        axs[4].set_ylim(ymin=0, ymax=1)
        axs[4].text(len(epsilons) - 1, epsilons[-1], str(epsilons[-1]))

        axs[5].set_ylabel('Its. T')
        axs[5].plot(iterations, 'k')
        axs[5].set_ylim(ymin=0)
        axs[5].text(len(iterations) - 1, iterations[-1], str(iterations[-1]))

        axs[6].set_ylabel('Its. M')
        axs[6].plot(iteration_foods, 'k')
        axs[6].set_ylim(ymin=0)
        axs[6].text(len(iteration_foods) - 1, iteration_foods[-1], str(iteration_foods[-1]))

        axs[6].set_xlabel('Number of Games')


        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(.1)
        # plt.close()

        # save model every 100 games
        if game_count % 100 == 0 and not use_checkpoint:
            directory = f"data/cnn/{timestamp}"
            os.makedirs(directory, exist_ok=True)
            torch.save({
                "epoch": game_count,
                "model_state_dict": game.model.state_dict(),
                "target_model_state_dict": game.target_model.state_dict(),
                "optimizer_state_dict": game.optimizer.state_dict(),
            }, f"{directory}/model_{game_count}.pth")