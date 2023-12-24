import datetime

import numpy as np
import itertools
import random
from deep_qlearning import *
import os
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    learning_rate: float = 0.7
    discount_factor: float = 0.5
    epsilon_min: float = 0.0
    epsilon_cutoff = 100
    game_cutoff = 500
    learning_type = LearningType.QLearningOffPolicy

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def run_game(hyperparameters, run_id=0, save_folder="data/", renderer=True):
    use_cached_q_values = False
    dump_cache = False
    os.makedirs(save_folder, exist_ok=True)

    if use_cached_q_values:
        q_values = np.load(f'{save_folder}/q_values.npy', allow_pickle=True).item()
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
    dump_every_n_games = 100
    desc = f"fromzero_ec{hyperparameters.epsilon_cutoff}_gc{hyperparameters.game_cutoff}_lr{hyperparameters.learning_rate}_df{hyperparameters.discount_factor}"
    filename = f"{save_folder}/deepqlearning_{desc}_{run_id}_{int(datetime.datetime.now().timestamp())}.txt"
    with open(filename, 'w') as f:
        f.write(f"Game,Score\n")
    while game_count < hyperparameters.game_cutoff:
        game = DeepQLearningSnakeGame(q_values, renderer)
        # if game_count % 100 == 0 and game_count > 90:
        #     game.epsilon *= 0.5
        game.learning_rate = hyperparameters.learning_rate
        game.discount_factor = hyperparameters.discount_factor
        game.learning_type = hyperparameters.learning_type

        if hyperparameters.epsilon_cutoff > 0:
            game.epsilon = max(hyperparameters.epsilon_min, 1.0 - (game_count / hyperparameters.epsilon_cutoff))
        else:
            game.epsilon = hyperparameters.epsilon_min

        game.frametime = 50_000
        game.block_size = 10
        game.play()
        game_count += 1
        print(f"Games: {game_count}, Score: {game.get_score()}")

        if dump_cache and game_count % dump_every_n_games == 0:
            np.save(f"{save_folder}/q_values.npy", q_values)

        with open(filename, 'a') as f:
            f.write(f"{game_count},{game.get_score()},{game.death_reason}\n")


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


if __name__ == "__main__":
    test_epsilons()
    # test_learning_types()
    # pure_test("new_distance_measurement")