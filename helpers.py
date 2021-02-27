import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def read_from_csv(fname: str):
    """Get data df with sorted columns."""
    df = pd.read_csv(fname, sep=';', index_col=0)
    df = df.reindex(columns=sorted(df.columns))
    return df


def discretize(df: pd.DataFrame, levels: int):
    """Discretize data into the a common number of gene-specific bins."""
    data_disc = np.zeros(df.shape)
    for i in range(df.shape[1]):
        vec = df.iloc[:,i].values
        bins = np.linspace(min(vec)-0.0001, max(vec)+0.0001, levels+1)
        data_disc[:,i] = np.digitize(vec,bins,right=False)
    data_disc = pd.DataFrame(data_disc.astype(int))
    data_disc.columns = df.columns
    data_disc.index = df.index
    return data_disc


def plot_data(df: pd.DataFrame, discr: bool):
    """Make plot of either raw or discretized measurement data."""
    plt.figure(figsize=(5,5))
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


def result_to_df(array: np.array):
    """Make array of edges into a neat data frame."""
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    df = pd.DataFrame(array.astype(int))
    df.columns=genes
    df.index=genes
    return df


def plot_result_directed(df: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix considering the directions."""
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(df.values, fignum=fig.number)
    plt.title('Dependency Matrix, directed\n(yellow at place (i,j) = edge from gene i to gene j)', fontsize=12)
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    plt.xticks(ticks=np.arange(0,5), labels=genes)
    plt.yticks(ticks=np.arange(0,5), labels=genes)
    return


def plot_result_undirected(df: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix excluding the directions."""
    df1 = df.values.copy()
    for i in range(5):
        for j in range(5):
            if df1[i,j] == 1:
                df1[j,i] = 1
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(df1, fignum=fig.number)
    plt.title('Dependency Matrix, undirected\n(yellow at place (i,j) = edge between genes i and j either way)', fontsize=12)
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    plt.xticks(ticks=np.arange(0,5), labels=genes)
    plt.yticks(ticks=np.arange(0,5), labels=genes)
    return


def calculate_rates_directed(df_to_try: pd.DataFrame, df_true: pd.DataFrame):
    """Calculate true positive and false positive rates."""
    df_to_try = df_to_try.values
    df_true = df_true.values
    n_truth_pos = np.sum(df_true)
    n_truth_neg = 25-n_truth_pos
    true_positives = np.sum((df_to_try==1) & (df_true==1))
    false_positives = np.sum((df_to_try==1) & (df_true==0))
    tpr = true_positives/n_truth_pos
    fpr = false_positives/n_truth_neg
    return tpr, fpr


def calculate_rates_undirected(df_to_try: pd.DataFrame, df_true: pd.DataFrame):
    """Calculate true positive and false positive rates."""
    df_to_try = df_to_try.values.copy()
    df_true = df_true.values.copy()
    for i in range(5):
        for j in range(5):
            if df_to_try[i,j] == 1:
                df_to_try[j,i] = 1
            if df_true[i,j] == 1:
                df_true[j,i] = 1
    n_truth_pos = np.sum(df_true)
    n_truth_neg = 25-n_truth_pos
    true_positives = np.sum((df_to_try==1) & (df_true==1))
    false_positives = np.sum((df_to_try==1) & (df_true==0))
    tpr = true_positives/n_truth_pos
    fpr = false_positives/n_truth_neg
    return tpr, fpr

    