import pandas as pd
import numpy as np
import random
from helpers import result_to_df


def correlation_model(df: pd.DataFrame, threshold: float):
    corr_matrix = df.corr().values
    result = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i==j:
                continue
            if corr_matrix[i, j] >= threshold:
                result[i, j] = 1
    return result_to_df(result)


def random_model(threshold=0.5, n_edges=0, n_rep=100):
    temp_matrix = np.zeros((5, 5))
    result = np.zeros((5, 5))
    edge_locations = []
    for i in range(5):
        for j in range(5):
            if i==j:
                continue
            edge_locations.append((i,j))
    if n_edges:  # number of edges specified
        for rep in range(n_rep):
            loc_edges = random.sample(edge_locations, n_edges)  # locations of edges
            for edge_loc in loc_edges:
                temp_matrix[edge_loc]+=1
        edge_counts = (-np.sort(-temp_matrix.flatten()))[:n_edges]
        n_added = 0
        for edge_count in np.unique(edge_counts):
            loc = np.where(temp_matrix==edge_count)
            for i in range(len(loc[0])):
                if n_added == n_edges:
                    break
                result[loc[0][i], loc[1][i]] = 1
                n_added += 1
    else:  # number of edges at random
        for i in range(5):
            for j in range(5):
                if i==j:
                    continue
                if random.random() >= threshold:
                    result[i, j] = 1
    return result_to_df(result)