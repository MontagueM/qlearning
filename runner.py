import datetime
import time

import numpy as np
import itertools
import random
from cnn_deep_qlearning import *
import os
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    epsilon_min: float = 0.0
    epsilon_cutoff = 200
    game_cutoff = 500
    learning_type = LearningType.QLearningOffPolicy
    batch_size = 32
    dimensions = (100, 100)
    device = "cpu"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def run_game(hyperparameters, run_id=0, save_folder="data/", renderer=True, time_allowed=0):
    os.makedirs(save_folder, exist_ok=True)

    game_count = 0
    desc = f"bs{hyperparameters.batch_size}_ec{hyperparameters.epsilon_cutoff}_gc{hyperparameters.game_cutoff}_lr{hyperparameters.learning_rate}_df{hyperparameters.discount_factor}"
    filename = f"{save_folder}/cnn_{desc}_{run_id}_{int(datetime.datetime.now().timestamp())}.txt"
    with open(filename, 'w') as f:
        f.write(f"Game,Score,DeathReason,LossMean,RewardMean,TimeS\n")

    start_time = time.time()
    agent = DDQNAgent(hyperparameters.device, 4)
    agent.epsilon_cutoff = hyperparameters.epsilon_cutoff
    agent.epsilon_min = hyperparameters.epsilon_min
    agent.learning_rate = hyperparameters.learning_rate
    agent.discount_factor = hyperparameters.discount_factor
    agent.batch_size = hyperparameters.batch_size

    while game_count < hyperparameters.game_cutoff:
        game = DeepQLearningSnakeGame(agent, renderer)

        game.dimensions = hyperparameters.dimensions

        agent.start_episode(game_count, game)

        game.frametime = 50_000
        game.block_size = 10
        game.play()
        game_count += 1

        game_data = agent.get_game_data(game)

        print(f"Games: {game_count}, Score: {game_data.score}, Epsilon: {agent.epsilon}")


        if time_allowed > 0 and time.time() - start_time > time_allowed:
            break

        with open(filename, 'a') as f:
            f.write(f"{game_count},{game_data.score},{game.death_reason},{np.mean(game_data.losses)},{np.mean(game_data.rewards)},{time.time()-start_time}\n")


def test_epsilons():
    epsilon_cutoffs = [*range(0, 20, 1), *range(20, 100, 10), *range(100, 1_001, 100)]
    reruns = 1
    time = int(datetime.datetime.now().timestamp())
    for epsilon_cutoff in epsilon_cutoffs:
        for index in range(reruns):
            print(f"Running epsilon cutoff {epsilon_cutoff} run {index+1}/{reruns}...")
            hyperparameters = Hyperparameters(epsilon_cutoff=epsilon_cutoff, game_cutoff=epsilon_cutoff+100, epsilon_min=0.0)
            run_game(hyperparameters, index, save_folder=f"data/epsilon_cutoff/{time}/{epsilon_cutoff}", renderer=False)


def test_learning_types():
    policy_types = [LearningType.QLearningOffPolicy, LearningType.SARSAOnPolicy]
    game_cutoff = 1_500
    reruns = 3
    time = int(datetime.datetime.now().timestamp())
    for policy_type in policy_types:
        for index in range(reruns):
            print(f"Running policy type {policy_type} run {index+1}/{reruns}...")
            hyperparameters = Hyperparameters(game_cutoff=game_cutoff, learning_type=policy_type)
            run_game(hyperparameters, index, save_folder=f"data/policy_type/{time}/{policy_type}", renderer=False)


def pure_test(desc):
    game_cutoff = 2_000
    hyperparameters = Hyperparameters(game_cutoff=game_cutoff)
    reruns = 3
    time = int(datetime.datetime.now().timestamp())
    for index in range(reruns):
        print(f"Running run {index+1}/{reruns}...")
        run_game(hyperparameters, index, save_folder=f"data/{desc}/{time}/main/", renderer=False)


def test_batches():
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    # batch_sizes = [2, 4]
    batch_sizes = [32, 64]
    batch_sizes = batch_sizes[::-1]
    game_cutoff = 2_000
    dimensions = (100, 100)
    reruns = 1
    time = int(datetime.datetime.now().timestamp())
    # time_allowed = 60 * 3
    time_allowed = 0
    for batch_size in batch_sizes:
        if batch_size >= 64:
            device = "mps"
        else:
            device = "cpu"
        torch.set_default_device(device)
        for index in range(reruns):
            print(f"Running batch size {batch_size} run {index+1}/{reruns}...")
            hyperparameters = Hyperparameters(game_cutoff=game_cutoff, batch_size=batch_size, dimensions=dimensions, device=device)
            run_game(hyperparameters, index, save_folder=f"data/batches/{time}/{batch_size}", renderer=True, time_allowed=time_allowed)


if __name__ == "__main__":
    # test_epsilons()
    # test_learning_types()
    # pure_test("new_distance_measurement")
    test_batches()