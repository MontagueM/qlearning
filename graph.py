import matplotlib.pyplot as plt
import os
import numpy as np

color_iter = iter(['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'brown', 'grey', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'tan', 'salmon', 'gold', 'lightcoral', 'darkkhaki', 'darkseagreen', 'darkslategray', 'darkolivegreen', 'darkslateblue', 'darkmagenta', 'darkred', 'darkorange', 'darkgreen', 'darkblue', 'darkcyan', 'darkviolet', 'darkgray', 'lightpink', 'lightsalmon', 'khaki', 'palegreen', 'lightsteelblue', 'thistle', 'lightgrey', 'violet', 'lightgoldenrodyellow', 'mediumorchid', 'slateblue', 'mediumvioletred', 'mediumturquoise', 'mediumspringgreen', 'mediumslateblue', 'mediumseagreen', 'mediumaquamarine', 'mediumblue', 'mediumslat'])


def plot_gamescores(all_scores, all_times):
    end_cutoff = 1

    fig, axs = plt.subplots(2, figsize=(6, 10), sharex=False)
    fig.subplots_adjust(top=0.95)

    diff_colour = {}

    for differentiator, game_scores in all_scores.items():
        colour = next(color_iter)
        diff_colour[differentiator] = colour
        x = list(game_scores.keys())[:-end_cutoff]
        y = [sum(scores)/len(scores) for scores in game_scores.values()][:-end_cutoff]
        axs[0].plot(x, y, 'x', color=colour, label=differentiator)
        # fill between 1 std dev
        # axs[0].fill_between(x,
        #                     [y[i] - np.std(scores) for i, scores in enumerate(list(game_scores.values())[:-end_cutoff])],
        #                     [y[i] + np.std(scores) for i, scores in enumerate(list(game_scores.values())[:-end_cutoff])], alpha=0.3, color=colour)
        # error bars, only the top/bottom bars
        axs[0].errorbar(x, y, yerr=[np.std(scores) for scores in list(game_scores.values())[:-end_cutoff]], fmt='none', ecolor=colour, capsize=3, elinewidth=0)

    axs[0].legend()

    # plt.fill_between
    axs[0].set_xlabel("Game")
    axs[0].set_ylabel("Score")
    axs[0].set_title("Score over time")

    # plot time instead
    ax1 = axs[1]
    for differentiator, game_scores in all_scores.items():
        x = [int(all_times[differentiator][int(i)-1]) if i > 0 else 0 for i in game_scores.keys()][:-end_cutoff]
        y = [sum(scores)/len(scores) for scores in game_scores.values()][:-end_cutoff]
        ax1.plot(x, y, 'x', color=diff_colour[differentiator], label=differentiator)

    ax1.set_ylabel("Score")
    ax1.set_xlabel("Time")

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

def plot_all(all_scores, all_deaths, all_losses, all_rewards):
    fig, axs = plt.subplots(4, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(top=0.95)

    colours = iter(["red", "blue", "green", "orange", "purple", "black"])

    for batch_size in all_scores.keys():
        colour = next(colours)
        scores = all_scores[batch_size]
        deaths = all_deaths[batch_size]
        losses = all_losses[batch_size]
        rewards = all_rewards[batch_size]

        axs[0].set_ylabel('Score')
        # axs[0].plot(scores, 'k')
        # running_trend on score
        # scores_mean = [np.mean(scores[i * running_trend:i * running_trend + running_trend]) for i in
        #                range(len(tail_deaths))]
        # axs[0].plot(running_trend_x, scores_mean, 'r')
        axs[0].scatter(list(scores.keys()), [sum(scores[game])/len(scores[game]) for game in scores.keys()], color=colour, label=batch_size, marker="x")
        # 1 std dev
        # axs[0].fill_between(running_trend_x,
        #                     [scores_mean[i] - np.std(scores[i * running_trend:i * running_trend + running_trend]) for i in
        #                      range(len(tail_deaths))],
        #                     [scores_mean[i] + np.std(scores[i * running_trend:i * running_trend + running_trend]) for i in
        #                      range(len(tail_deaths))], alpha=0.3, color="red")
        # axs[0].set_ylim(ymin=0)
        # axs[0].text(len(scores) - 1, scores[-1], str(scores[-1]))

        # axs[1].set_ylabel('Loss')
        # axs[1].plot(_losses, 'k')
        # axs[1].set_ylim(ymin=0)
        # axs[1].text(len(losses) - 1, losses[-1], str(losses[-1]))
        #
        # axs[2].set_ylabel('Rew.M')
        # axs[2].plot(_rewards_mean, 'k')
        # axs[2].set_ylim(ymin=0)
        # axs[2].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
        #
        # axs[3].set_ylabel('Death')
        # axs[3].plot(running_trend_x, tail_deaths, 'x', label="Tail", color="red")
        # axs[3].plot(running_trend_x, wall_deaths, 'x', label="Wall", color="blue")
        # axs[3].plot(running_trend_x, loop_deaths, 'x', label="Loop", color="green")
        # axs[3].set_ylim(ymin=0, ymax=1)
        # axs[3].legend(loc="upper left")

        axs[3].set_xlabel('Number of Games')

    plt.legend()
    plt.savefig(f'{top_folder}/plot_all.png', dpi=400)
    plt.show()

if __name__ == '__main__':
    # top_folder = 'data/epsilon_cutoff/1703003251'
    # top_folder = 'data/policy_type/1703035101'
    # top_folder = 'data/new_distance_measurement/1703091383'
    # top_folder = "data/batches/1704054346"  # 3 minute time limit
    # top_folder = "data/batches/1704055473"  # 1000 game limit
    top_folder = "data/batches/1704059217"  # only 32 and 64 for 2k games
    game_variants = {}

    all_scores = {}
    all_deaths = {}
    all_losses = {}
    all_rewards = {}
    all_times = {}
    for folder in os.listdir(top_folder):
        if os.path.isfile(f'{top_folder}/{folder}'):
            continue
        cutoff = folder
        all_scores[cutoff] = {}
        all_deaths[cutoff] = {}
        all_losses[cutoff] = {}
        all_rewards[cutoff] = {}
        all_times[cutoff] = []
        # each are re-runs for a mean calculation
        for file in os.listdir(f'{top_folder}/{folder}'):
            filename = f'{top_folder}/{folder}/{file}'
            print(filename)

            with open(filename, 'r') as f:
                lines = f.readlines()


            for line in lines[1:]:
                game, score, death_reason, loss, reward, time = line.split(',')

                game = int(game)
                score = int(score)

                # bin scores into x
                bin_size = len(lines) // 20

                new_game = (game // bin_size) * bin_size
                if new_game not in all_scores[cutoff]:
                    all_scores[cutoff][new_game] = []
                    all_deaths[cutoff][new_game] = []
                    all_losses[cutoff][new_game] = []
                    all_rewards[cutoff][new_game] = []
                all_scores[cutoff][new_game].append(score)
                all_deaths[cutoff][new_game].append(death_reason.strip().split(".")[-1])
                all_losses[cutoff][new_game].append(float(loss))
                all_rewards[cutoff][new_game].append(float(reward))
                all_times[cutoff].append(float(time))


    plot_gamescores(all_scores, all_times)
    # plot_deaths(all_deaths)
    # plot_all(all_scores, all_deaths, all_losses, all_rewards)