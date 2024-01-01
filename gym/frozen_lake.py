import time

import gymnasium as gym

from dataclasses import dataclass

from gymnasium.core import RenderFrame

from cnn_deep_qlearning import weights_init, GameData
from exit.exit_game import ExitGame, Agent
import torch
import torch.nn as nn
import numpy as np
from misc import Direction, Coordinate
from copy import deepcopy
from typing import Tuple, List
import cv2
from collections import deque
import random
import matplotlib.pyplot as plt

from general import AbstractGame


@dataclass
class Experience:
    state: torch.Tensor
    action: Direction
    reward: float
    _step: int
    log_action_probabilities: torch.Tensor
    terminated: bool


@dataclass
class EpisodeData:
    episode_num: int
    epsilons: List[float]
    rewards: List[float]
    losses: List[float]
    steps: int


class Plotter:
    def __init__(self):
        self.env_state = None
        self.nn_state = None
        self.render_image = None
        self.episode_data: List[EpisodeData] = []

        self.fig, self.axs = plt.subplots(4, figsize=(6, 10), sharex=False)

    def render(self):
        self.__render_states()
        self.__render_graphs()

        plt.pause(0.00001)
        plt.draw()

    def __render_states(self):
        # upscale to rgb array size
        # duplicate to rgb channels
        nn_state = self.nn_state.repeat(3, 1, 1)
        nn_state = cv2.resize(nn_state.numpy().transpose(1, 2, 0), (self.env_state.shape[1], self.env_state.shape[0]),
                              interpolation=cv2.INTER_AREA)

        # combine the two side-by-side
        final_screen = np.concatenate((self.env_state, nn_state), axis=1)
        final_screen = final_screen.astype(np.uint8)

        # display rbg array
        if self.render_image is None:
            self.render_image = self.axs[0].imshow(final_screen)
        else:
            self.render_image.set_data(final_screen)

    def __render_graphs(self):
        axs = self.axs[1]
        axs.set_ylabel('s/e')
        axs.plot([e.steps for e in self.episode_data], 'k')

        axl = self.axs[2]
        axl.set_ylabel('loss')
        axl.plot([np.mean(e.losses) for e in self.episode_data], 'k')

        axr = self.axs[3]
        axr.set_ylabel('reward')
        axr.plot([np.mean(e.rewards) for e in self.episode_data], 'k')


class REINFORCEGymAgent(nn.Module, Agent):
    def __init__(self, device, num_actions, env: gym.Env, plotter: Plotter):
        super(REINFORCEGymAgent, self).__init__()
        self.device = device
        self.num_actions = num_actions
        self.plotter = plotter

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(6),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=1)
        )
        self.model.to(self.device)
        self.model.apply(weights_init)

        self.target_model = deepcopy(self.model)
        self.target_model_update = 10_00

        self.learning_rate = 0
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
        self.current_env_steps = 0
        self.grid_since_last_food = None
        self.loop_count = 0

        self.env: gym.Env = env

    def start_episode(self, episode_num):
        self.epsilon = max(self.epsilon_min, 1.0 - episode_num / self.epsilon_cutoff) if self.epsilon_cutoff > 0 else 0
        self.epsilons.append(self.epsilon)
        self.current_env_steps = 0
        self.experience_memory = []
        print(f"starting episode {episode_num}, epsilon: {self.epsilon}")

    def end_episode(self):
        self.train()

    def act(self) -> bool:
        state = self.get_state()
        action, log_action_probabilities = self.get_action(state)

        observation, reward, terminated, truncated, info = self.env.step(action)

        if reward == 0:
            if terminated:
                reward = -1
            else:
                reward = -0.1
        next_state = self.get_state()

        self.current_env_steps += 1
        self.total_steps += 1

        # death is complicated; we set the "previous" state to death so we actually register it.
        # setting a future state as death doesnt do anything as there is no next state to update
        # we care about being 1 step before death + action that causes death, which is done in the training.
        if terminated:
            self.game_end_indices.append(self.total_steps)

        if self.total_steps % self.target_model_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("Updated target model")

        self.rewards.append(reward)

        self.experience_memory.append(
            Experience(state, action, reward, _step=self.total_steps, log_action_probabilities=log_action_probabilities,
                       terminated=terminated))

        return terminated

    def get_action(self, state: torch.Tensor) -> Tuple[Direction, torch.Tensor]:
        action_tensor = self.model(state.unsqueeze(0))
        log_action_probabilities = torch.log(action_tensor)
        # fix nan
        # log_action_probabilities[log_action_probabilities.isnan()] = -1000
        # choose based on probability of softmax
        print(action_tensor)
        action = torch.multinomial(action_tensor, 1)
        action = action.item()

        return action, log_action_probabilities

    def get_state(self) -> torch.Tensor:
        return self.get_grid()

    def get_grid(self) -> torch.Tensor:
        render_frame: RenderFrame = env.render()
        self.plotter.env_state = render_frame

        # convert to grayscale
        render_frame = cv2.cvtColor(render_frame, cv2.COLOR_RGB2GRAY)
        # downsample
        render_frame = cv2.resize(render_frame, (20, 20), interpolation=cv2.INTER_AREA)

        grid = torch.tensor(render_frame, dtype=torch.float32).unsqueeze(0)
        # flip width and height
        # grid = grid.transpose(1, 2)

        self.plotter.nn_state = grid
        return grid

    def get_reward(self, terminated: torch.Tensor) -> float:
        if terminated:
            reward = 1
        else:
            reward = 0
        return reward

    def train(self) -> None:
        # -1 because we need the next state, -1 because the next state should have a reward(?)

        total_loss = 0
        total_reward = 0
        for t, h in enumerate(self.experience_memory):
            empirical_discounted_reward = 0
            for k in range(t + 1, len(self.experience_memory)):
                empirical_discounted_reward += self.discount_factor ** (k - t - 1) * self.experience_memory[k].reward

            total_reward += empirical_discounted_reward
            if total_reward > 0:
                pass
            policy_loss = h.log_action_probabilities[0][h.action]
            total_loss += policy_loss.item()

            self.optimizer.zero_grad()
            print(policy_loss)
            gradients_wrt_params(self.model, policy_loss)
            update_params(self.model, self.learning_rate * self.discount_factor ** t * empirical_discounted_reward)

        self.losses.append(total_loss)
        self.rewards.append(total_reward)

    def get_game_data(self) -> GameData:
        game_epsilons = self.epsilons[-self.current_env_steps:]
        game_rewards = self.rewards[-self.current_env_steps:]
        game_losses = self.losses[-self.current_env_steps:]
        game_steps = self.current_env_steps
        return GameData(game_epsilons, game_rewards, game_losses, game_steps)


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


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='rgb_array')

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError('Expected discrete action space')

    plotter = Plotter()
    agent = REINFORCEGymAgent("cpu", env.action_space.n, env, plotter)
    agent.epsilon_cutoff = 0
    episode_count_max = 10000
    episode_count = 0
    while episode_count < episode_count_max:
        env.reset()
        agent.start_episode(episode_count)

        terminated = False
        while not terminated:
            terminated = agent.act()
            plotter.render()

        # time.sleep(0.1)
        agent.end_episode()
        episode_count += 1

        game_data = agent.get_game_data()
        episode_data = EpisodeData(episode_count, game_data.epsilons, game_data.rewards, game_data.losses, game_data.steps)
        plotter.episode_data.append(episode_data)

        plotter.render()

    env.close()
