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

from cnn_deep_qlearning import GameData, State
from snake_game import AbstractSnakeGame, Coordinate, DeathReason, Agent
from misc import Direction, Vector
from dataclasses import dataclass
import torch, torch.nn as nn

import matplotlib.pyplot as plt


plt.ion()


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

global_count = 0
losses = []
rewards = []


@dataclass
class Experience:
    state: State
    action: Direction
    reward: float
    _step: int
    log_action_probabilities: torch.Tensor

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

class REINFORCEAgent(nn.Module, Agent):
    def __init__(self, device, num_actions):
        super(REINFORCEAgent, self).__init__()
        self.device = device
        self.num_actions = num_actions

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(6),
            nn.Flatten(),
            nn.Linear(32*6*6, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=1)
        )
        self.model.to(self.device)
        self.model.apply(weights_init)

        self.target_model = deepcopy(self.model)
        self.target_model_update = 10_00

        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_cutoff = 200
        self.memory_size = 100_000
        self.experience_memory: list[Experience] = []

        self.batch_size = 32

        self.training_start = 1_000

        self.losses = []
        self.rewards = []
        self.game_end_indices = []
        self.epsilons = []

        self.do_train = True

        self.total_steps = 0
        self.current_game_steps = 0
        self.grid_since_last_food = None
        self.loop_count = 0

    def start_episode(self, episode_num, game):
        self.epsilon = max(self.epsilon_min, 1.0 - episode_num / self.epsilon_cutoff)
        self.epsilons.append(self.epsilon)
        self.current_game_steps = 0
        self.experience_memory = []
        self.grid_since_last_food = np.zeros((game.dimensions[0] // game.block_size, game.dimensions[1] // game.block_size))
        print(f"starting episode {episode_num}, epsilon: {self.epsilon}")

    def act(self, game: AbstractSnakeGame) -> None:
        state = self.get_state(game)
        action, log_action_probabilities = self.get_action(state)

        game.move_direction = action
        game.move_snake()
        game.update()

        self.current_game_steps += 1
        self.total_steps += 1

        next_state = self.get_state(game)

        # death is complicated; we set the "previous" state to death so we actually register it.
        # setting a future state as death doesnt do anything as there is no next state to update
        # we care about being 1 step before death + action that causes death, which is done in the training.
        state.died = not game.snake_alive
        if state.died:
            self.game_end_indices.append(self.total_steps)

        if self.total_steps % self.target_model_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("Updated target model")

        reward = self.get_reward(state, next_state, game)
        self.rewards.append(reward)

        self.experience_memory.append(Experience(state, action, reward, _step=self.total_steps, log_action_probabilities=log_action_probabilities))

    def get_action(self, state: State) -> Tuple[Direction, torch.Tensor]:
        action_tensor = self.model(state.grid.unsqueeze(0))

        log_action_probabilities = torch.log(action_tensor)
        # choose based on probability of softmax
        action = torch.multinomial(action_tensor, 1)
        action = action.item()
        action = Direction(action)

        return action, log_action_probabilities

    def get_state(self, game: AbstractSnakeGame) -> State:
        distance_to_food = game.food_location - game.snake[0]
        return State(game.food_location, distance_to_food, self.get_grid(game))

    def get_grid(self, game: AbstractSnakeGame) -> torch.Tensor:
        # convert stateful info into 2d grid with 1 channel, its 3 but combine for efficiency
        grid = np.zeros((game.dimensions[0] // game.block_size, game.dimensions[1] // game.block_size, 3))
        # food is green (0.5)
        grid[game.food_location.x // game.block_size, game.food_location.y // game.block_size, 1] = 1
        # snake head is red (0.25)
        grid[game.snake[0].x // game.block_size, game.snake[0].y // game.block_size, 0] = 1
        # snake body is white (1)
        for body_part in game.snake[1:]:
            grid[body_part.x // game.block_size, body_part.y // game.block_size, :] = 1
        # add walls
        for wall in game.walls:
            grid[wall.x // game.block_size, wall.y // game.block_size, :] = 0.5

        # permute to fit BATCH x CHANNELS x HEIGHT x WIDTH
        grid = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1)
        return grid

    def get_reward(self, state: State, next_state: State, game: AbstractSnakeGame) -> float:
        if state.died:
            reward = -5
        elif next_state.food_position != state.food_position:
            reward = 3
            self.grid_since_last_food = np.zeros((game.dimensions[0] // game.block_size, game.dimensions[1] // game.block_size))
        elif abs(next_state.distance_to_food.x) < abs(state.distance_to_food.x) or abs(next_state.distance_to_food.y) < abs(state.distance_to_food.y):
            reward = 1
        else:
            reward = -1

        detect_loops = True
        if detect_loops:
            snake_head = game.snake[0]
            if self.grid_since_last_food[snake_head.x // game.block_size, snake_head.y // game.block_size] == 1:
                # reward = -1
                self.loop_count += 1

                if not self.do_train and self.loop_count > 100:# and self.epsilon < 0.1:
                    game.snake_alive = False
                    game.death_reason = DeathReason.LOOP

            self.grid_since_last_food[snake_head.x // game.block_size, snake_head.y // game.block_size] = 1

        return reward

    def train(self, game) -> None:
        # -1 because we need the next state, -1 because the next state should have a reward(?)

        total_loss = 0
        total_reward = 0
        for t, h in enumerate(self.experience_memory):
            empirical_discounted_reward = 0
            for k in range(t+1, len(self.experience_memory)):
                empirical_discounted_reward += self.discount_factor ** k * self.experience_memory[k].reward

            total_reward += empirical_discounted_reward
            self.optimizer.zero_grad()
            policy_loss = h.log_action_probabilities[0][h.action.value]
            total_loss += policy_loss.item()
            gradients_wrt_params(self.model, policy_loss)
            update_params(self.model, self.learning_rate * self.discount_factor ** t * empirical_discounted_reward)

        self.losses.append(total_loss)
        self.rewards.append(total_reward)

    def get_game_data(self, game: AbstractSnakeGame) -> GameData:
        game_epsilons = self.epsilons[-self.current_game_steps:]
        game_rewards = self.rewards[-self.current_game_steps:]
        game_losses = self.losses[-self.current_game_steps:]
        game_score = game.get_score()
        return GameData(game_epsilons, game_rewards, game_losses, game_score)

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

    # mps seems to be better at large batch sizes e.g. 1024-2048
    dev_name = "cpu"
    # dev_name = "mps"
    torch.set_default_device(dev_name)
    device = torch.device(dev_name)
    # torch.set_default_device("cpu")
    # device = torch.device("cpu")

    # with open(filename, 'w') as f:
    #     f.write(f"Game,Score\n")

    timestamp = int(datetime.datetime.now().timestamp())
    use_checkpoint = False
    checkpoint_file = "data/cnn/1703695105/model_1700.pth"
    action_every_n_frames = 1

    agent = REINFORCEAgent(device, 4)

    while game_count < game_count_cap:
        reinforce_game = AbstractSnakeGame(agent, True)

        # randomise dimensions between 50-250, for now always square
        dim = random.randint(5, 10) * 10 * 2
        reinforce_game.dimensions = (dim, dim)
        print(f"Dimensions: {reinforce_game.dimensions}")
        reinforce_game.dimensions = (100, 100)
        if game_count == 0:
            num_params = sum(p.numel() for p in agent.model.parameters())
            print(f"Number of parameters: {num_params}, mean weight: {sum(p.sum() for p in agent.model.parameters()) / num_params}")

            if use_checkpoint:
                checkpoint = torch.load(checkpoint_file, map_location=device)
                agent.model.load_state_dict(checkpoint["model_state_dict"])
                agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                print(f"New mean weight: {sum(p.sum() for p in agent.model.parameters()) / num_params}")


        if use_checkpoint:
            # todo figure out why setting this to False causes the game to stop working
            # my guess is that something in the model is not initialized properly
            agent.do_train = False

        reinforce_game.frametime = 50_000
        if use_checkpoint:
            reinforce_game.frametime = 50

        agent.start_episode(game_count, reinforce_game)
        reinforce_game.play()
        agent.train(reinforce_game)
        game_data = agent.get_game_data(reinforce_game)
        if not game_data.losses:
            game_data.losses.append(0)
        game_count += 1
        print(f"Games: {game_count}, Score: {game_data.score}, Epsilon: {agent.epsilon}, Loss: {game_data.losses[-1]}")

        # with open(filename, 'a') as f:
        #     f.write(f"{game_count},{game.get_score()},{game.death_reason}\n")

        scores.append(game_data.score)
        _losses.append(np.mean(game_data.losses))
        _rewards_sum.append(np.sum(game_data.rewards))
        _rewards_mean.append(np.mean(game_data.rewards))
        death_reasons.append(reinforce_game.death_reason)
        iterations.append(agent.current_game_steps / game_data.score if game_data.score != 0 else agent.current_game_steps)

        if game_count % running_trend == 0:
            # proportion of deaths
            tail_deaths.append(death_reasons.count(DeathReason.TAIL) / len(death_reasons))
            wall_deaths.append(death_reasons.count(DeathReason.WALL) / len(death_reasons))
            loop_deaths.append(death_reasons.count(DeathReason.LOOP) / len(death_reasons))
            death_reasons = []

        num_almost_zero_weights = 0
        threshold = 1e-3
        for name, param in agent.model.named_parameters():
            num_almost_zero_weights += torch.sum(torch.abs(param) < threshold).item()

        sparsity = num_almost_zero_weights / sum(p.numel() for p in agent.model.parameters())
        # print(f"Sparsity: {sparsity}")


        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.close('all')

        plt.clf()

        fig, axs = plt.subplots(6, figsize=(5, 10), sharex=True)
        fig.subplots_adjust(top=0.95)
        if use_checkpoint:
            fig.suptitle(f"Inference on {checkpoint_file}")
        else:
            fig.suptitle(f'Train {timestamp}: LR={agent.learning_rate}, BS={agent.batch_size} DIMS={reinforce_game.dimensions[0]}x{reinforce_game.dimensions[1]}')

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
        axs[1].text(len(game_data.losses) - 1, game_data.losses[-1], str(game_data.losses[-1]))

        # axs[2].set_ylabel('Rew.S')
        # axs[2].plot(_rewards_sum, 'k')
        # # running_trend on reward sum
        # rewards_sum_mean = [np.mean(_rewards_sum[i*running_trend:i*running_trend + running_trend]) for i in range(len(tail_deaths))]
        # axs[2].plot(running_trend_x, rewards_sum_mean, 'r')
        # axs[2].set_ylim(ymin=0)
        # axs[2].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))

        axs[2].set_ylabel('Rew.M')
        axs[2].plot(_rewards_mean, 'k')
        axs[2].set_ylim(ymin=0)
        axs[2].text(len(game_data.rewards) - 1, game_data.rewards[-1], str(game_data.rewards[-1]))

        axs[3].set_ylabel('Death')
        axs[3].plot(running_trend_x, tail_deaths, 'x', label="Tail", color="red")
        axs[3].plot(running_trend_x, wall_deaths, 'x', label="Wall", color="blue")
        axs[3].plot(running_trend_x, loop_deaths, 'x', label="Loop", color="green")
        axs[3].set_ylim(ymin=0, ymax=1)
        axs[3].legend(loc="upper left")

        axs[4].set_ylabel('Eps')
        axs[4].plot(agent.epsilons, 'k')
        axs[4].set_ylim(ymin=0, ymax=1)
        axs[4].text(len(agent.epsilons) - 1, agent.epsilons[-1], str(agent.epsilons[-1]))

        axs[5].set_ylabel('Its. T')
        axs[5].plot(iterations, 'k')
        axs[5].set_ylim(ymin=0)
        axs[5].text(len(iterations) - 1, iterations[-1], str(iterations[-1]))
        #
        # axs[6].set_ylabel('Its. M')
        # axs[6].plot(iteration_foods, 'k')
        # axs[6].set_ylim(ymin=0)
        # axs[6].text(len(iteration_foods) - 1, iteration_foods[-1], str(iteration_foods[-1]))

        axs[5].set_xlabel('Number of Games')


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
                "model_state_dict": agent.model.state_dict(),
                "target_model_state_dict": agent.target_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
            }, f"{directory}/model_{game_count}.pth")