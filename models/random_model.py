import pandas as pd
import numpy as np
import random
from .helpers import result_to_df


def random_model(threshold=0.5, n_edges=0):
    """No threshold is needed here."""
    result = np.zeros((5, 5))
    edge_locations = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            edge_locations.append((i, j))
    if n_edges:  # number of edges specified
        loc_edges = random.sample(
            edge_locations, n_edges)  # locations of edges
        for edge_loc in loc_edges:
            result[edge_loc] = 1
    else:  # number of edges at random
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                if random.random() >= threshold:
                    result[i, j] = 1
    return result_to_df(result)
