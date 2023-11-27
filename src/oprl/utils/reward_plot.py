import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
sns.set_style('whitegrid')
colors = ['greyish', 'faded blue', "faded green"]
sns.set_palette(sns.xkcd_palette(colors))


def load_xy(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    data = np.asarray(data)
    return data[:, 1:]


def plot_data(paths, output_path, n_rows=1, n_cols=3, smooth_len=5, lw=3):
    figure(num=0, figsize=(20, 5), dpi=100, facecolor='w', edgecolor='k')
    for i_subplot, env_name in enumerate(paths):
        ax = plt.subplot(n_rows, n_cols, i_subplot + 1)
        plt.title(env_name)

        for model_name in paths[env_name]:
            data = load_xy(paths[env_name][model_name])

            # Smooth reward
            for i in range(data.shape[0] - 1, smooth_len, -1):
                data[i, 1] = np.mean(data[i - smooth_len:i, 1])

            data = pd.DataFrame(data, columns=['step', 'reward'])
            data['model'] = model_name
            sns.lineplot(data=data, x='step', y='reward', lw=lw)

        ax.legend(labels=list(paths[env_name]), loc='lower right')

    plt.savefig(output_path)


if __name__ == "__main__":
    paths = {
        'Pendulum-v0': {
            'D3PG': '~/pendulum_d3pg.json',
            'D4PG': '~/pendulum_d4pg.json'
        },
        'LunarLanderContinuous-v2': {
            'D3PG': '~/lunar_d3pg.json',
            'D4PG': '~/lunar_d4pg.json'
        },
        'BipedalWalker-v2': {
            'D3PG': '~/bipedal_d3pg.json',
            'D4PG': '~/bipedal_d4pg.json'
        }
    }
    plot_data(paths, "plot.png", n_rows=1, n_cols=3)