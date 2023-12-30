from typing import Type

from PIL import Image
from dataclasses import dataclass
from ale_py import ALEInterface, SDL_SUPPORT
from ale_py.roms import Breakout
import torch
import torch.nn as nn

import random
from collections import deque
import numpy as np
import cv2

import matplotlib.pyplot as plt

import abc

# torch.set_default_device("mps")
# device = torch.device("mps")
torch.set_default_device("cpu")
device = torch.device("cpu")

class Agent(abc.ABC):

    def __init__(self, num_actions):
        self.num_actions = num_actions

    @abc.abstractmethod
    def get_action(self, ale: ALEInterface):
        pass

    @abc.abstractmethod
    def train(self, reward, died):
        pass

    @abc.abstractmethod
    def start_episode(self, episode_num):
        pass


class RandomAgent(Agent):
    def __init__(self, num_actions):
        super(RandomAgent, self).__init__(num_actions)

    def get_action(self, ale: ALEInterface):  # dependency injection
        return random.randrange(self.num_actions)


@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float = 0.0
    died: bool = False

def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform(layer_in.weight)
        layer_in.bias.data.fill_(0.0)


class DQNAgent(nn.Module, Agent):
    def __init__(self, num_actions):
        super(DQNAgent, self).__init__()
        self.num_actions = num_actions

        self.frames_per_state = 4  # enough to see the ball move (number from papers)
        self.frame_skip = 4  # also from paper
        self.channels_per_frame = 1  # greyscale

        self.model = nn.Sequential(
            nn.Conv2d(self.frames_per_state * self.channels_per_frame, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.model.to(device)

        self.model.apply(weights_init)

        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_length = 2_0
        self.memory_size = 10_000
        self.replay_memory: deque[Experience] = deque(maxlen=self.memory_size)

        self.batch_size = 32

        self.training_start = 5_000

    def start_episode(self, episode_num):
        self.epsilon = max(self.epsilon_min, 1.0 - episode_num / self.epsilon_length)
        print(f"starting episode {episode_num}, epsilon: {self.epsilon}")

    def get_action(self, ale: ALEInterface):
        # grayscale the screen
        screen = ale.getScreenGrayscale()
        # downsampling
        # screen = screen[::6, ::8]
        # screen = screen[::3, ::2]

        # 0 = 0, >0 = 1
        screen = np.where(screen > 0, 255, 0)
        # uint8
        screen = screen.astype(np.uint8)

        screen = cv2.resize(screen[25:-15, :], (80, 84), interpolation=cv2.INTER_AREA)

        # im = Image.fromarray(screen)
        # im.save("test.png")

        screen = torch.from_numpy(screen).float()

        # return randrange(self.num_actions)
        if random.random() < self.epsilon or len(self.replay_memory) < self.training_start:
            action = random.randrange(self.num_actions)
        else:
            # get past 3 frames and stack
            state = torch.stack([screen, *[self.replay_memory[-i].state for i in range(self.frames_per_state - 1)]])
            action = self.model(state.unsqueeze(0)).argmax().item()

        self.replay_memory.append(Experience(screen, action))

        return action

    def train(self, reward, died):
        # save the reward
        self.replay_memory[-1].reward = reward if not died else -1
        self.replay_memory[-1].died = died


        if len(self.replay_memory) < self.training_start:
            return

        # sample a batch
        if len(self.replay_memory)-1 < self.batch_size:
            batch_indices = range(len(self.replay_memory)-1)
        else:
            batch_indices = random.sample(range(len(self.replay_memory)-1), self.batch_size)  # -1 because we need the next state

        batch = [self.replay_memory[i] for i in batch_indices]

        # get states - each state is frame N, N-1, N-2, N-3
        # states becomes a tensor of shape (batch_size, frames_per_state, height, width)
        states = torch.stack([torch.stack([self.replay_memory[i-j].state for j in range(self.frames_per_state)]) for i in batch_indices])
        actions = torch.tensor([experience.action for experience in batch])
        rewards = torch.tensor([experience.reward for experience in batch])
        dead = torch.tensor([exp.died for exp in batch])
        # this is invalid if the last state is a dead state, but we don't actually use it due to the dead mask
        next_states = torch.stack([torch.stack([self.replay_memory[i-j+1].state for j in range(self.frames_per_state)]) for i in batch_indices])

        # 1 if alive, 0 if dead
        dead_mask = dead.logical_not().float()

        states = states.to(device)
        next_states = next_states.to(device)

        predicted = self.model(states)
        y = predicted.clone()

        q = torch.max(self.model(next_states), dim=1).values
        r = torch.arange(len(batch))
        y[r, actions] = rewards + self.discount_factor * q * dead_mask

        self.optimizer.zero_grad()
        loss = self.loss_fn(predicted, y)
        loss_float = loss.item()
        loss.backward()
        self.optimizer.step()

        return loss_float


class BreakoutGame:
    def __init__(self, agent_type: Type[Agent], render=False, seed=123):
        self.ale = ALEInterface()

        self.render = render
        if self.render and SDL_SUPPORT:
            self.ale.setBool("sound", False)
            self.ale.setBool("display_screen", True)

        self.ale.setInt("random_seed", seed)

        self.ale.setInt("frame_skip", 4)

        self.ale.loadROM(Breakout)
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.legal_actions)
        self.agent = agent_type(self.num_actions)

        self.render_image = None

    def do_render(self):
        if self.render:
            pass

        rgb_array = self.ale.getScreenRGB()

        # combine with the screen the CNN sees
        screen = self.agent.replay_memory[-1].state
        # upscale to rgb array size
        # duplicate to rgb channels
        screen = screen.unsqueeze(0).repeat(3, 1, 1)
        screen = cv2.resize(screen.numpy().transpose(1, 2, 0), (rgb_array.shape[1], rgb_array.shape[0]), interpolation=cv2.INTER_AREA)

        # combine the two side-by-side
        final_screen = np.concatenate((rgb_array, screen), axis=1)
        final_screen = final_screen.astype(np.uint8)

        # display rbg array
        if self.render_image is None:
            self.render_image = plt.imshow(final_screen)
        else:
            self.render_image.set_data(final_screen)

        plt.pause(0.00001)
        plt.draw()


    def play(self, num_episodes=10):
        for episode in range(num_episodes):
            total_reward = 0
            self.agent.start_episode(episode)
            while not self.ale.game_over():
                a = self.agent.get_action(self.ale)
                # Apply an action and get the resulting reward
                reward = self.ale.act(a)
                self.agent.train(reward, self.ale.game_over())
                self.do_render()
                total_reward += reward
            print(f"Episode {episode} ended with score: {total_reward}")
            self.ale.reset_game()


if __name__ == "__main__":
    game = BreakoutGame(DQNAgent, render=False)
    game.play(num_episodes=10_000)
