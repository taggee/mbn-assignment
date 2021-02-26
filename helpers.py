import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def result_to_df(array: np.array):
    """Make array of edges into a neat data frame."""
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    df = pd.DataFrame(array.astype(int))
    df.columns=genes
    df.index=genes
    return df


def read_from_csv(fname: str):
    """Get data df with sorted columns."""
    df = pd.read_csv(fname, sep=';', index_col=0)
    df = df.reindex(columns=sorted(df.columns))
    return df


def plot_result(df: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix."""
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(df.values, fignum=fig.number)
    plt.title('Dependency Matrix', fontsize=12)
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    plt.xticks(ticks=np.arange(0,5), labels=genes)
    plt.yticks(ticks=np.arange(0,5), labels=genes)
    return
