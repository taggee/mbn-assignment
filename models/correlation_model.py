import pandas as pd
import numpy as np
from .helpers import result_to_df, calculate_rates_directed, calculate_rates_undirected


def correlation_model(df: pd.DataFrame, threshold: float):
    corr_matrix = df.corr().values
    result = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            if corr_matrix[i, j] >= threshold:
                result[i, j] = 1
    return result_to_df(result)


def correlation_model_roc_values(df_data: pd.DataFrame, df_truth: pd.DataFrame,
                                 threshold_min: float, threshold_max: float):
    """Calculate tpr's and fpr's for ROC curve of the correlation model."""
    
    # initialize:
    thresholds = np.linspace(threshold_min, threshold_max, 20)
    tpr_dir = np.zeros(len(thresholds))
    fpr_dir = np.zeros(len(thresholds))
    tpr_undir = np.zeros(len(thresholds))
    fpr_undir = np.zeros(len(thresholds))
    
    # calcluate directed and undirected tpr and fpr values:
    for i, thres in enumerate(thresholds):
        mat = correlation_model(df=df_data, threshold=thres)
        tpr_dir[i], fpr_dir[i] = calculate_rates_directed(df_to_try=mat, df_true=df_truth)
        tpr_undir[i], fpr_undir[i] = calculate_rates_undirected(df_to_try=mat, df_true=df_truth)

    # return a text string and the directed and undirected tpr and fpr values:
    string = 'Correlation model, \n thresholds {:.2f} to {:.2f}'.format(threshold_min, threshold_max)
    return [string, tpr_dir, fpr_dir, tpr_undir, fpr_undir]
