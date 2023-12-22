import datetime
import itertools
import random
from typing import List, Tuple
from enum import Enum
from collections import deque

import numpy as np

from snake_game import AbstractSnakeGame, Coordinate, DeathReason
from misc import Direction, Vector
from dataclasses import dataclass
import torch, torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib

from IPython import display

plt.ion()
# matplotlib.use('TkAgg')


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

    def tensor(self):
        return torch.tensor([float(self.relative_food_direction[0]), float(self.relative_food_direction[1]), *[float(x) for x in list(self.surroundings)]])


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


@dataclass
class LearningType(Enum):
    QLearningOffPolicy = 0
    SARSAOnPolicy = 1
    FittedQLearning = 2


class DeepQLearningLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # def forward(self, discount_factor, reward, q1, q0):
    #     # instead of getting q from the q_values, we get it from the model
    #     # todo this model is wrong??, it predicts action not q - actually maybe these are the same?
    #     loss = reward + discount_factor * q1 - q0
    #     lsq_loss = loss ** 2
    #     # print(f"loss: {lsq_loss}, q1 {q1}, q0 {q0}")
    #     return lsq_loss

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

        return lsq_loss.mean()

def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

global_count = 0
losses = []
rewards = []


class DeepQLearningSnakeGame(AbstractSnakeGame):
    # define neural net (https://github.com/blakeMilner/DeepQLearning/blob/master/deepqlearn.lua)
    # states and actions that go into the neural net (state0,action0),(state1,action1), ... , (stateN)
    # this variable controls the size of the temporal window

    # Number of past state/action pairs input to the network. 0 = agent lives in-the-moment :)
    sqs = [''.join(s) for s in list(itertools.product(*[['0', '1', '2']] * 4))]
    widths = ['0', '1', '4']
    heights = ['2', '3', '4']

    num_states = len(sqs) * len(widths) * len(heights)

    num_states = 6  # 4 for surroundings, 2 for food position
    num_actions = 4
    temporal_window = 2
    # net_inputs = (num_states + num_actions) * temporal_window + num_states
    net_inputs = num_states * temporal_window
    num_inputs = num_states
    hidden_nodes_1 = 256
    hidden_nodes_2 = hidden_nodes_1

    train_start = 5000
    batch_size = 32
    mem_size = 100_000

    learning_rate = 0.00025
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    eps_steps = 100_000

    replay_memory = deque(maxlen=mem_size)

    target_model_update = 10_000

    target_model = nn.Sequential(
        nn.Linear(num_inputs, hidden_nodes_1),
        nn.ReLU(),
        nn.Linear(hidden_nodes_1, hidden_nodes_2),
        nn.ReLU(),
        nn.Linear(hidden_nodes_2, num_actions)
    )

    model = nn.Sequential(
        nn.Linear(num_inputs, hidden_nodes_1),
        nn.ReLU(),
        nn.Linear(hidden_nodes_1, hidden_nodes_2),
        nn.ReLU(),
        nn.Linear(hidden_nodes_2, num_actions)
    )

    # He initialization of weights

    target_model.apply(weights_init)
    model.apply(weights_init)

    # set loss function
    # criterion = DeepQLearningLoss()
    criterion = nn.MSELoss()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def __init__(self, state_dict=None, use_renderer=True):
        super().__init__(use_renderer)
        self.frametime = 50

        self.history: List[History] = []

        self.learning_type = LearningType.QLearningOffPolicy

        self.local_count = 0


    def get_action(self) -> Direction:
        state = self.get_state()

        # epsilon-greedy todo how does this work with model inference?
        if random.random() < self.epsilon or global_count < self.train_start:
            action = random.randint(0, 3)
            action_direction = Direction(action)
            self.history.append(History(state, action_direction))
            return action_direction

        # NN forward pass
        # state to tensor
        state_tensor = state.tensor()

        # prev_state_tensor = self.history[-1].state.tensor()
        # if prev_state is not None:
        #
        # else:
        #     state_tensor_prev = torch.zeros_like(state_tensor)

        # concat prev state and current state
        # state_tensor_all = torch.cat((prev_state_tensor, state_tensor))
        action_tensor = self.model(state_tensor)
        action = action_tensor.argmax().item()

        # take index of max value
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
        elif abs(current.state.distance_to_food.x) < abs(previous.state.distance_to_food.x) or abs(current.state.distance_to_food.y) < abs(previous.state.distance_to_food.y):  # Snake is closer to the food, positive reward
            reward = 1
        # if snake next to tail, negative reward
        else:
            reward = -1  # Snake is further from the food, negative reward

        # discourage going back to where you came from to avoid oscillation
        # if len(self.history) > 2:
        #     sm1 = self.history[-3].state  # state before previous state
        #     if sm1 == h1.state:
        #         reward += -2

        # if s0.surroundings[0] == '2' or s0.surroundings[1] == '2' or s0.surroundings[2] == '2' or s0.surroundings[3] == '2':
        #     reward += -1

        rewards.append(reward)

        if not self.snake_alive:
            reward = -5
            self.replay_memory.append(
                Experience(current.state, current.action, float(reward), current.state, not self.snake_alive))
        else:
            self.replay_memory.append(Experience(previous.state, previous.action, float(reward), current.state, not self.snake_alive))

        loss = self.train()
        losses.append(loss)

        self.local_count += 1
        global_count += 1

        if global_count % self.target_model_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if global_count % 5_000 == 0:
            print(f"Global count: {global_count}")

    def train(self) -> float:
        if global_count < self.train_start:
            return 0

        batch = random.sample(self.replay_memory, self.batch_size)
        rewards = torch.tensor([exp.reward for exp in batch])
        if any([exp.dead for exp in batch]):
            print("dead")
        if any([exp.reward > 1 for exp in batch]):
            print("reward")
        states = torch.stack([exp.state.tensor() for exp in batch])
        actions = torch.tensor([exp.action.value for exp in batch])
        # one-hot encode actions as all action values are equally valuable, e.g. 3 is not better than 2
        # actions_onehot = self.get_onehot(actions)
        next_states = torch.stack([exp.next_state.tensor() for exp in batch])
        dead = torch.tensor([exp.dead for exp in batch])

        # clipped_error = -1.0 * error.clamp(-1, 1)


        # loss = self.criterion(self.target_model, self.discount_factor, rewards, states, actions, next_states, dead)
        # todo replace with above but for now using MSE on the calculated (actual) vs the models output q value
        # the model should learn what rewards map to what states, learn the discount factor, so the only thing
        # we need to give it should be the previous state and the current state. Given this the model should be discriminated
        # to evaluate an approximation of q which is what our equation does.
        # state_and_next_state = torch.cat((states, next_states), dim=1)
        # predicted = self.model(states)

        q0_actions = self.target_model(states)
        batch_indices = torch.arange(len(actions))
        q0 = q0_actions[batch_indices, actions]
        q1 = torch.max(self.target_model(next_states), dim=1).values

        # discount_factor_tensor = torch.full((self.batch_size, 4), self.discount_factor, dtype=torch.float32)
        # discount_factor_tensor = torch.where(dead, torch.zeros_like(discount_factor_tensor), discount_factor_tensor)

        # expected = predicted.clone()
        q0 = self.model(states)
        q1 = self.model(next_states)

        # update each action with the expected q value, leave the rest the same as did not change
        delta = np.zeros((self.batch_size, self.num_actions))
        delta[np.arange(self.batch_size), actions] = 1
        delta = torch.tensor(delta, dtype=torch.float32)

        dead = dead.repeat_interleave(self.num_actions).reshape(self.batch_size, self.num_actions)
        rewards = rewards.repeat_interleave(self.num_actions).reshape(self.batch_size, self.num_actions)

        # expected[batch_indices, actions] = rewards + discount_factor_tensor * q1
        targets = (1 - delta) * q0 + delta * (rewards + ~dead * self.discount_factor * q1)

        loss = self.criterion(q0, targets)
        loss_float = loss.item()

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return loss_float

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.batch_size, 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot


if __name__ == "__main__":
    game_count = 0
    game_count_cap = 20000
    epsilon_trigger = 100
    desc = f"fromzero_e{epsilon_trigger}"
    filename = f"qlearning_{desc}_{int(datetime.datetime.now().timestamp())}_{game_count_cap}.txt"
    state_dict = None
    scores = []
    _losses = []
    _rewards_sum = []
    _rewards_mean = []
    tails = []
    walls = []
    death_reasons = []
    running_trend = 50
    epsilons = []
    with open(filename, 'w') as f:
        f.write(f"Game,Score\n")
    while game_count < game_count_cap:
        game = DeepQLearningSnakeGame(state_dict, False)
        # if game_count % 100 == 0 and game_count > 90:
        #     game.epsilon *= 0.5
        # if game_count > epsilon_trigger:
        #     game.epsilon = 0
        game.epsilon = max(game.epsilon_min, 1.0 - (global_count / game.eps_steps))
        epsilons.append(game.epsilon)
        game.frametime = 50
        game.block_size = 10
        rewards = []
        losses = []
        game.play()
        game_count += 1
        print(f"Games: {game_count}, Score: {game.get_score()}, Epsilon: {game.epsilon}")
        state_dict = game.model.state_dict()

        with open(filename, 'a') as f:
            f.write(f"{game_count},{game.get_score()},{game.death_reason}\n")

        scores.append(game.get_score())
        _losses.append(np.mean(losses))
        _rewards_sum.append(np.sum(rewards))
        _rewards_mean.append(np.mean(rewards))
        death_reasons.append(game.death_reason)

        if game_count % running_trend == 0:
            # proportion of deaths
            tails.append(death_reasons.count(DeathReason.TAIL) / len(death_reasons))
            walls.append(death_reasons.count(DeathReason.WALL) / len(death_reasons))
            death_reasons = []


        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.clf()

        fig, axs = plt.subplots(6, figsize=(5, 10), sharex=True)
        fig.suptitle('Training...')

        running_trend_x = [(1+i) * running_trend for i in range(len(tails))]

        axs[0].set_ylabel('Score')
        axs[0].plot(scores, 'k')
        # running_trend on score
        scores_mean = [np.mean(scores[i*running_trend:i*running_trend + running_trend]) for i in range(len(tails))]
        axs[0].plot(running_trend_x, scores_mean, 'r')
        axs[0].set_ylim(ymin=0)
        axs[0].text(len(scores) - 1, scores[-1], str(scores[-1]))

        axs[1].set_ylabel('Loss')
        axs[1].plot(_losses, 'k')
        axs[1].set_ylim(ymin=0)
        axs[1].text(len(losses) - 1, losses[-1], str(losses[-1]))

        axs[2].set_ylabel('Rew.S')
        axs[2].plot(_rewards_sum, 'k')
        # running_trend on reward sum
        rewards_sum_mean = [np.mean(_rewards_sum[i*running_trend:i*running_trend + running_trend]) for i in range(len(tails))]
        axs[2].plot(running_trend_x, rewards_sum_mean, 'r')
        axs[2].set_ylim(ymin=0)
        axs[2].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))

        axs[3].set_ylabel('Rew.M')
        axs[3].plot(_rewards_mean, 'k')
        axs[3].set_ylim(ymin=0)
        axs[3].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))

        axs[4].set_ylabel('Death')
        axs[4].plot(running_trend_x, tails, 'x', label="Tail", color="red")
        axs[4].plot(running_trend_x, walls, 'x', label="Wall", color="blue")
        axs[4].set_ylim(ymin=0, ymax=1)
        axs[4].legend(loc="upper right")

        axs[5].set_ylabel('Eps')
        axs[5].plot(epsilons, 'k')
        axs[5].set_ylim(ymin=0, ymax=1)
        axs[5].text(len(scores) - 1, epsilons[-1], str(epsilons[-1]))


        axs[5].set_xlabel('Number of Games')


        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(.1)
        plt.close()