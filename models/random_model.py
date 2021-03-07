import pandas as pd
import numpy as np
import random
from .helpers import result_to_df, calculate_rates_directed, calculate_rates_undirected


def random_model(threshold=0.5, n_edges=0):
    """No threshold is needed if n_edges is set."""
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


def random_model_roc_values(df_truth: pd.DataFrame, threshold_min: float, 
                            threshold_max: float, repetitions: int):
    """Calculate tpr's and fpr's for ROC curve of the random model by averaging over several repetitions."""
    
    # initialize:
    thresholds = np.linspace(threshold_min, threshold_max, 20)
    tpr_dir = np.zeros(len(thresholds))
    fpr_dir = np.zeros(len(thresholds))
    tpr_undir = np.zeros(len(thresholds))
    fpr_undir = np.zeros(len(thresholds))
    
    # calcluate directed tpr and fpr values:
    for i, thres in enumerate(thresholds):
        tpr_temp = np.zeros(repetitions)
        fpr_temp = np.zeros(repetitions)
        for rep in range(repetitions):
            mat = random_model(threshold=thres)
            tpr_temp[rep],fpr_temp[rep] = calculate_rates_directed(df_to_try=mat, df_true=df_truth)
        tpr_dir[i], fpr_dir[i] = np.mean(tpr_temp), np.mean(fpr_temp)

    # calcluate undirected tpr and fpr values:
    for i, thres in enumerate(thresholds):
        tpr_temp = np.zeros(repetitions)
        fpr_temp = np.zeros(repetitions)
        for rep in range(repetitions):
            mat = random_model(threshold=thres)
            tpr_temp[rep],fpr_temp[rep] = calculate_rates_undirected(df_to_try=mat, df_true=df_truth)
        tpr_undir[i], fpr_undir[i] = np.mean(tpr_temp), np.mean(fpr_temp)

    # return a text string and the directed and undirected tpr and fpr values:
    string = 'Random model, \n thresholds {:.2f} to {:.2f}, {:d} repetitions each'.format(threshold_min, threshold_max, repetitions)
    return [string, tpr_dir, fpr_dir, tpr_undir, fpr_undir, thresholds]
