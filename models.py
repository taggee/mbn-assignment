import pandas as pd
import numpy as np


def correlation_model(df, threshold):
    corr_matrix = df.corr().values
    result = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if corr_matrix[i, j] >= threshold:
                result[i, j] = 1
    return pd.DataFrame(result)
