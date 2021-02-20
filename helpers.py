import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def read_from_csv(fname: str):
    """Get data df with sorted columns."""
    df = pd.read_csv(fname, sep=';', index_col=0)
    df = df.reindex(columns=sorted(df.columns))
    return df


def plot_array(array: np.array):
    """Plot the 5x5 gene dependency matrix."""
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(array, fignum=fig.number)
    plt.title('Dependency Matrix', fontsize=12)
    return fig
