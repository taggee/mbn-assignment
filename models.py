import pandas as pd
import numpy as np
from helpers import result_to_df


def correlation_model(df: pd.DataFrame, threshold: float):
    corr_matrix = df.corr().values
    result = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if corr_matrix[i, j] >= threshold:
                result[i, j] = 1
    return result_to_df(result)
