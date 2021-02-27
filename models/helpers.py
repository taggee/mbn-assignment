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
        vec = df.iloc[:, i].values
        bins = np.linspace(min(vec)-0.0001, max(vec)+0.0001, levels+1)
        data_disc[:, i] = np.digitize(vec, bins, right=False)
    data_disc = pd.DataFrame(data_disc.astype(int))
    data_disc.columns = df.columns
    data_disc.index = df.index
    return data_disc


def result_to_df(array: np.array):
    """Make array of edges into a neat data frame."""
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    df = pd.DataFrame(array.astype(int))
    df.columns = genes
    df.index = genes
    return df


def calculate_rates_directed(df_to_try: pd.DataFrame, df_true: pd.DataFrame):
    """Calculate true positive and false positive rates."""
    df_to_try = df_to_try.values
    df_true = df_true.values
    n_truth_pos = np.sum(df_true)
    n_truth_neg = 25-n_truth_pos
    true_positives = np.sum((df_to_try == 1) & (df_true == 1))
    false_positives = np.sum((df_to_try == 1) & (df_true == 0))
    tpr = true_positives/n_truth_pos
    fpr = false_positives/n_truth_neg
    return tpr, fpr


def calculate_rates_undirected(df_to_try: pd.DataFrame, df_true: pd.DataFrame):
    """Calculate true positive and false positive rates."""
    df_to_try = df_to_try.values.copy()
    df_true = df_true.values.copy()
    for i in range(5):
        for j in range(5):
            if df_to_try[i, j] == 1:
                df_to_try[j, i] = 1
            if df_true[i, j] == 1:
                df_true[j, i] = 1
    n_truth_pos = np.sum(df_true)
    n_truth_neg = 25-n_truth_pos
    true_positives = np.sum((df_to_try == 1) & (df_true == 1))
    false_positives = np.sum((df_to_try == 1) & (df_true == 0))
    tpr = true_positives/n_truth_pos
    fpr = false_positives/n_truth_neg
    return tpr, fpr
