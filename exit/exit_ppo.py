from dataclasses import dataclass

from cnn_deep_qlearning import weights_init, GameData
from exit.exit_game import ExitGame, Agent
import torch
import torch.nn as nn
import numpy as np
from misc import Direction, Coordinate
from copy import deepcopy
from typing import Tuple, List
from collections import deque
import random
import matplotlib.pyplot as plt

from general import AbstractGame


@dataclass
class State:
    exit_position: Coordinate  # only used to know if snake ate food
    grid: torch.Tensor
    complete: bool = False


@dataclass
class Experience:
    state: State
    action: Direction
    reward: float
    _step: int
    log_action_probabilities: torch.Tensor

class REINFORCEExitAgent(nn.Module, Agent):
    def __init__(self, device, num_actions):
        super(REINFORCEExitAgent, self).__init__()
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

        self.learning_rate = 2**-13
        self.discount_factor = 0.99
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
        self.epsilon = max(self.epsilon_min, 1.0 - episode_num / self.epsilon_cutoff) if self.epsilon_cutoff > 0 else 0
        self.epsilons.append(self.epsilon)
        self.current_game_steps = 0
        self.experience_memory = []
        self.grid_since_last_food = np.zeros((game.dimensions[0] // game.block_size, game.dimensions[1] // game.block_size))
        print(f"starting episode {episode_num}, epsilon: {self.epsilon}")

    def act(self, game: ExitGame) -> None:
        state = self.get_state(game)
        action, log_action_probabilities = self.get_action(state)

        game.move_direction = action
        game.act()

        self.current_game_steps += 1
        self.total_steps += 1

        next_state = self.get_state(game)

        # death is complicated; we set the "previous" state to death so we actually register it.
        # setting a future state as death doesnt do anything as there is no next state to update
        # we care about being 1 step before death + action that causes death, which is done in the training.
        state.complete = game.at_exit
        if state.complete:
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

    def get_state(self, game: ExitGame) -> State:
        return State(game.food_location, self.get_grid(game))

    def get_grid(self, game: ExitGame) -> torch.Tensor:
        # convert stateful info into 2d grid with 1 channel, its 3 but combine for efficiency
        grid = np.zeros((game.dimensions[0] // game.block_size, game.dimensions[1] // game.block_size, 3))
        # food is green (0.5)
        grid[game.exit_position.x // game.block_size, game.exit_position.y // game.block_size, 1] = 1
        # snake head is white (0.25)
        grid[game.agent_position.x // game.block_size, game.agent_position.y // game.block_size, :] = 1
        # add walls
        for wall in game.walls:
            grid[wall.x // game.block_size, wall.y // game.block_size, :] = 0.5

        # permute to fit BATCH x CHANNELS x HEIGHT x WIDTH
        grid = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1)
        return grid


    def get_reward(self, state: State, next_state: State, game: ExitGame) -> float:
        if state.complete:
            reward = 1
        else:
            reward = 0
        return reward

    def train(self, game) -> None:
        # -1 because we need the next state, -1 because the next state should have a reward(?)

        total_loss = 0
        total_reward = 0
        for t, h in enumerate(self.experience_memory):
            empirical_discounted_reward = 0
            for k in range(t+1, len(self.experience_memory)):
                empirical_discounted_reward += self.discount_factor ** (k - t - 1) * self.experience_memory[k].reward

            total_reward += empirical_discounted_reward
            policy_loss = h.log_action_probabilities[0][h.action.value]
            total_loss += policy_loss.item()

            self.optimizer.zero_grad()
            gradients_wrt_params(self.model, policy_loss)
            update_params(self.model, self.learning_rate * self.discount_factor ** t * empirical_discounted_reward)


        self.losses.append(total_loss)
        self.rewards.append(total_reward)

    def get_game_data(self) -> GameData:
        game_epsilons = self.epsilons[-self.current_game_steps:]
        game_rewards = self.rewards[-self.current_game_steps:]
        game_losses = self.losses[-self.current_game_steps:]
        game_score = self.current_game_steps
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

    dev_name = "cpu"
    # dev_name = "mps"
    torch.set_default_device(dev_name)
    device = torch.device(dev_name)

    agent = REINFORCEExitAgent(device, 4)
    agent.epsilon_cutoff = 0
    agent.epsilon_min = 0.01
    game_count = 0
    game_count_cap = 20_000
    while game_count < game_count_cap:
        egame = ExitGame(agent, False)
        egame.frametime = 50_000
        # egame.dimensions = (200, 200)
        agent.start_episode(game_count, egame)
        egame.play()
        agent.train(egame)

        game_data = agent.get_game_data()
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
        iterations.append(agent.current_game_steps / game_data.score if game_data.score != 0 else agent.current_game_steps)

        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.close('all')

        plt.clf()

        fig, axs = plt.subplots(6, figsize=(5, 10), sharex=True)
        fig.subplots_adjust(top=0.95)
        fig.suptitle(f'Train: LR={agent.learning_rate}, BS={agent.batch_size} DIMS={egame.dimensions[0]}x{egame.dimensions[1]}')

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

        axs[2].set_ylabel('Rew.M')
        axs[2].plot(_rewards_mean, 'k')
        # axs[2].set_ylim(ymin=0)
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

        axs[5].set_xlabel('Number of Games')


        fig.canvas.draw()
        fig.canvas.flush_events()
