import pandas as pd
import numpy as np
import random
from itertools import combinations
from .helpers import (
    result_to_df,
    discretize,
    calculate_rates_directed,
    calculate_rates_undirected,
)
from sklearn.linear_model import LinearRegression
from typing import List
from itertools import chain


def _edges_with(edge_list: List[tuple], gene_idx: int):
    """Find the edges the gene at gene_idx has an edge with."""
    edges_with_gene = [[x, y] for x, y in edge_list if x == gene_idx or y == gene_idx]
    [x.remove(gene_idx) for x in edges_with_gene]
    return list(chain(*edges_with_gene))


def calculate_model_score(df: pd.DataFrame, edges: List[tuple]):
    """Calculate score with gradient matching."""
    scores = np.zeros(5)
    data_arr = df.to_numpy()
    for gene_idx in range(5):
        edge_partners = _edges_with(edges, gene_idx)
        # Calculate target values with gradient matching
        dividend = data_arr[1:, gene_idx] - data_arr[:-1, gene_idx]
        divisor = (df.index[1:] - df.index[:-1]).T
        y = dividend / divisor[:, np.newaxis]
        if edge_partners:
            X = data_arr[1:, edge_partners]
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            scores[gene_idx] = ((y - y_pred) ** 2).sum()
        else:
            y_pred = np.ndarray(shape=(19, 1))
            y_pred[:, 0] = y.mean()
            scores[gene_idx] = ((y - y_pred) ** 2).sum()
    return scores.mean()


def get_next_model(df: pd.DataFrame, edge_indexes: List[tuple]):
    """Return the next best edge and the model score:

    Params:
            gene_df
            locations of edges
    """
    edge_network = np.zeros(shape=(df.shape[1], df.shape[1]))
    model_score = 1
    for idx, _ in np.ndenumerate(edge_network):
        if idx[0] == idx[1] or (idx in edge_indexes):
            # edges with same parent and child not allowed
            continue
        new_model_score = calculate_model_score(df, edge_indexes + [idx])
        if new_model_score <= model_score:
            model_score = new_model_score
            best_edge = idx
    return model_score, best_edge


def greedy_hillclimbing(df: pd.DataFrame, iterations: int):
    """Run greedy hillclimbing gene network inference."""
    edge_network = np.zeros(shape=(df.shape[1], df.shape[1]))
    edge_indexes = []
    model_scores = []
    for _ in range(iterations):
        model_score, best_edge = get_next_model(df, edge_indexes)
        edge_indexes.append(best_edge)
        model_scores.append(model_score)
        edge_network[best_edge[0], best_edge[1]] = 1
    return result_to_df(edge_network)


def hillclimbing_model_roc_values(
    df_data: pd.DataFrame,
    df_truth: pd.DataFrame,
    iterations_min: int,
    iterations_max: int,
):
    """Calculate tpr's and fpr's for ROC curve of the Greedy hillclimbing model."""

    # initialize:
    iters = np.arange(iterations_min, iterations_max + 1)
    tpr_dir = np.zeros(len(iters))
    fpr_dir = np.zeros(len(iters))
    tpr_undir = np.zeros(len(iters))
    fpr_undir = np.zeros(len(iters))

    # calcluate directed and undirected tpr and fpr values:
    for i, iteration in enumerate(iters):
        mat = greedy_hillclimbing(
            df=df_data,
            iterations=iteration,
        )
        tpr_dir[i], fpr_dir[i] = calculate_rates_directed(
            df_to_try=mat, df_true=df_truth
        )
        tpr_undir[i], fpr_undir[i] = calculate_rates_undirected(
            df_to_try=mat, df_true=df_truth
        )

    # return a text string and the directed and undirected tpr and fpr values
    string = f"Greedy hillclimbing model, \n number of added edges {iterations_min} to {iterations_max}"
    return [string, tpr_dir, fpr_dir, tpr_undir, fpr_undir, iters]
