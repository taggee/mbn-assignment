import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(df: pd.DataFrame, discr: bool):
    """Make plot of either raw or discretized measurement data."""
    plt.figure(figsize=(5, 5))
    plt.plot(df.index.values, df.values, "o-")
    plt.xlabel('time (min)')
    plt.legend(df.columns.values)
    if discr:
        plt.ylabel('gene expression\n(discretized units, low to high expression)')
        plt.yticks(ticks=np.unique(df.values), labels=np.unique(df.values))
        plt.title('Gene expression data, discretized')
    else:
        plt.ylabel('gene expression (arbitrary raw units)')
        plt.title('Gene expression data, raw')
    plt.show()
    return


def plot_result_directed(df: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix considering the directions."""
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(df.values, fignum=fig.number)
    plt.title(
        'Dependency Matrix, directed\n(yellow at place (i,j) = edge from gene i to gene j)', fontsize=12)
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    plt.xticks(ticks=np.arange(0, 5), labels=genes)
    plt.yticks(ticks=np.arange(0, 5), labels=genes)
    return


def plot_result_undirected(df: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix excluding the directions."""
    df1 = df.values.copy()
    for i in range(5):
        for j in range(5):
            if df1[i, j] == 1:
                df1[j, i] = 1
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(df1, fignum=fig.number)
    plt.title('Dependency Matrix, undirected\n(yellow at place (i,j) = edge between genes i and j either way)', fontsize=12)
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    plt.xticks(ticks=np.arange(0, 5), labels=genes)
    plt.yticks(ticks=np.arange(0, 5), labels=genes)
    return
