# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:13:18 2021

@author: A
"""

# antti will contribute here...

import numpy as np
import pandas as pd
import os

# os.chdir('C:\\Users\\A\\OneDrive - Aalto University\\Aalto\\MBN 5III\\Assignment\\mbn-assignment')
file = open('gene-data.csv')
data = file.readlines()
file.close()
# aakkosjärjestykseen?
gene_names = sorted(data[0].strip().split(';')[1:])
# ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']


ground_truth = np.array([[0,1,0,0,0],
                         [0,0,1,0,0],
                         [0,0,0,0,1],
                         [0,0,0,0,0],
                         [1,1,0,1,0]])
# ground_truth[i,j]=1 -> directed edge between gene i and gene j


def correlation_model (DATA, threshold):
    corr_matrix = pd.dataFrame(DATA).corr().values
    result = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if corr_matrix[i,j] >= threshold:
                result[i,j] = 1
    return result


    
    



