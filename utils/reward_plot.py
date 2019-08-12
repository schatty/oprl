import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import pickle
from glob import glob


def plot_rewards(dirname, figsize=(10, 5), dpi=150):
    """
    Plot rewards from all agents.

    Args:
        dirname (str): directory with .pkl logs.
    """
    # Read pickle logs
    file_paths = glob(f"{dirname}/*agent*.pkl")
    files = []
    for fp in file_paths:
        with open(fp, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            data = unpickler.load()
            files.append(data)

    # Create dataframe
    reward_index = files[0]['tasks'].index('reward')
    df = pd.concat([pd.Series(log_pkl['results'][reward_index])
                    for log_pkl in files], axis=1)
    df.columns = [f'reward_{i}' for i in range(len(files))]
    df['episode'] = df.index

    # Plot reward from each agent-process
    figure(num=0, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    sns.set_style('whitegrid')
    for i in range(len(file_paths)):
        sns.lineplot(data=df, y=f'reward_{i}', x='episode')
    plt.legend(title='Reward', loc='upper left',
               labels=[f"Process {i}" for i in range(len(files))])

    fn = f"{dirname}/reward_plot.png"
    plt.savefig(fn)