import pandas as pd
import numpy as np
from itertools import combinations
from .helpers import result_to_df, discretize


def aracne_model(df: pd.DataFrame, discr_bins: int, threshold: float, remove=True):
    """Threshold is for the minimum mutual information acceptable for an edge."""

    def rel_frequencies_single(gene_array: np.array, values: np.array):
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            result[i] = np.count_nonzero(gene_array == value)
        return result/len(gene_array)

    def rel_frequencies_pair(gene_array1: np.array, gene_array2: np.array, values: np.array):
        result = np.zeros((len(values), len(values)))
        for i, value1 in enumerate(values):
            for j, value2 in enumerate(values):
                result[i, j] = np.count_nonzero(
                    (gene_array1 == value1) & (gene_array2 == value2))
        return result/len(gene_array1)

    # calculate mutual information between all gene pairs:
    data_disc = discretize(df, discr_bins).values
    disc_values = np.arange(discr_bins)+1
    mi_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(i+1, 5):
            p_x = rel_frequencies_single(data_disc[:, i], disc_values)[
                :, np.newaxis]
            p_y = rel_frequencies_single(data_disc[:, j], disc_values)[
                :, np.newaxis]
            p_xy = rel_frequencies_pair(
                data_disc[:, i], data_disc[:, j], disc_values)
            mi = 0
            for i1, value1 in enumerate(disc_values):
                for j1, value2 in enumerate(disc_values):
                    if p_xy[i1, j1] == 0:
                        continue
                    mi += p_xy[i1, j1]*np.log(p_xy[i1, j1]/(p_x[i1]*p_y[j1]))
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # create an edge between genes that have a large enough MI value:
    result = np.zeros((5, 5))
    edge_locations = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            if mi_matrix[i, j] >= threshold:
                result[i, j] = 1
                edge_locations.append((i, j))

    # discard the weakest link from each triplet of edges independently of the others:
    if not remove:
        return result_to_df(result)
    edges_to_remove = []
    for triplet_loc in list(combinations(edge_locations, 3)):
        mi_values = np.array([mi_matrix[triplet_loc[0]],
                              mi_matrix[triplet_loc[1]],
                              mi_matrix[triplet_loc[2]]])
        smallest_loc = np.argmin(mi_values)
        edges_to_remove.append(triplet_loc[smallest_loc])
    for edge_loc in edges_to_remove:
        result[edge_loc] = 0
    return result_to_df(result)
