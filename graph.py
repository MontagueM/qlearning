import matplotlib.pyplot as plt
import os
import numpy as np

color_iter = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])


def plot_gamescores(all_scores):
    end_cutoff = 1
    for differentiator, game_scores in all_scores.items():
        x = list(game_scores.keys())[:-end_cutoff]
        y = [sum(scores)/len(scores) for scores in game_scores.values()][:-end_cutoff]
        colour = next(color_iter)
        plt.plot(x, y, 'x', color=colour, label=differentiator)
        plt.legend()

        # plt.fill_between
        plt.xlabel("Game")
        plt.ylabel("Score")
        plt.title("Score over time")
    # save 4k res
    plt.savefig(f'{top_folder}/plot_score.png', dpi=400)
    plt.show()


def plot_deaths(all_deaths):
    labels = True
    for _, game_deaths in all_deaths.items():
        x = list(game_deaths.keys())
        # tabulate deaths
        deaths = {}
        for game, death_reasons in game_deaths.items():
            deaths[game] = {}
            for death_reason in death_reasons:
                if death_reason not in deaths[game]:
                    deaths[game][death_reason] = 0
                deaths[game][death_reason] += 1

        # plot
        death_reason_colours = {"TAIL": "red", "WALL": "blue"}
        for i, (death_reason, colour) in enumerate(death_reason_colours.items()):
            # y-axis as proportion of deaths
            death_prop = [(deaths[game].get(death_reason, 0)) / sum(deaths[game].values()) for game in deaths]
            plt.plot(x, death_prop, 'x', color=colour, label=death_reason if labels else None)
        labels = False


    plt.legend()

    # plt.fill_between
    plt.xlabel("Game")
    plt.ylabel("Death proportion")
    plt.title("Deaths over time")
    # save 4k res
    plt.savefig(f'{top_folder}/plot_death.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    # top_folder = 'data/epsilon_cutoff/1703003251'
    top_folder = 'data/policy_type/1703035101'
    game_variants = {}

    all_scores = {}
    all_deaths = {}
    for folder in os.listdir(top_folder):
        if os.path.isfile(f'{top_folder}/{folder}'):
            continue
        cutoff = folder
        all_scores[cutoff] = {}
        all_deaths[cutoff] = {}
        # each are re-runs for a mean calculation
        for file in os.listdir(f'{top_folder}/{folder}'):
            filename = f'{top_folder}/{folder}/{file}'
            print(filename)

            with open(filename, 'r') as f:
                lines = f.readlines()


            for line in lines[1:]:
                game, score, death_reason = line.split(',')

                game = int(game)
                score = int(score)

                # bin scores into x
                bin_size = 50

                new_game = (game // bin_size) * bin_size
                if new_game not in all_scores[cutoff]:
                    all_scores[cutoff][new_game] = []
                    all_deaths[cutoff][new_game] = []
                all_scores[cutoff][new_game].append(score)
                all_deaths[cutoff][new_game].append(death_reason.strip().split(".")[-1])

    plot_gamescores(all_scores)
    plot_deaths(all_deaths)