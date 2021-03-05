import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(df: pd.DataFrame, discr: bool):
    """Make plot of either raw or discretized measurement data."""
    plt.figure(figsize=(5, 5))
    plt.plot(df.index.values, df.values, "o-")
    plt.xlabel("time (min)")
    plt.legend(df.columns.values)
    if discr:
        plt.ylabel("gene expression\n(discretized units, low to high expression)")
        plt.yticks(ticks=np.unique(df.values), labels=np.unique(df.values))
        plt.title("Gene expression data, discretized")
    else:
        plt.ylabel("gene expression (arbitrary raw units)")
        plt.title("Gene expression data, raw")
    plt.show()
    return


def make_result_mat(df1, df2):
    for idx, _ in np.ndenumerate(df2):
        if df2[idx] == 1:
            df2[idx[1], idx[0]] = 1
    result_mat = np.zeros(shape=(5, 5), dtype=int)
    result_mat[:] = 3
    for i in range(5):
        for j in range(5):
            if df1[i, j] == df2[i, j] and df1[i, j] == 1:
                result_mat[j, i] = 0
                result_mat[i, j] = 0
            elif df1[i, j] > df2[i, j]:
                result_mat[j, i] = 1
                result_mat[i, j] = 1
            elif df1[i, j] < df2[i, j]:
                result_mat[j, i] = 2
                result_mat[i, j] = 2
    return result_mat


def plot_result_directed(df: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix considering the directions."""
    fig = plt.figure(figsize=(5, 5))
    plt.matshow(df.values, fignum=fig.number)
    plt.title(
        "Dependency Matrix, directed\n(yellow at place (i,j) = edge from gene i to gene j)",
        fontsize=12,
    )
    genes = ["ASH1", "CBF1", "GAL4", "GAL80", "SWI5"]
    plt.xticks(ticks=np.arange(0, 5), labels=genes)
    plt.yticks(ticks=np.arange(0, 5), labels=genes)
    return


def plot_result_undirected(model_df: pd.DataFrame, ground_truth: pd.DataFrame):
    """Plot the 5x5 gene dependency matrix excluding the directions."""
    result_mat = make_result_mat(model_df.values, ground_truth.values)
    fig, ax = plt.subplots(1, 1)
    cmap = plt.get_cmap("Accent", np.max(result_mat) - np.min(result_mat) + 1)
    # set limits .5 outside true range
    mat = ax.matshow(
        result_mat,
        cmap=cmap,
        vmin=np.min(result_mat) - 0.5,
        vmax=np.max(result_mat) + 0.5,
    )
    # tell the colorbar to tick at integers
    cbar = fig.colorbar(
        mat, ticks=np.arange(np.min(result_mat), np.max(result_mat) + 1)
    )
    cbar.ax.set_yticklabels(["True Positive", "False Positive", "Not found", "No edge"])
    ax.set_title(
        "Dependency Matrix",
        fontsize=12,
    )
    genes = ["", "ASH1", "CBF1", "GAL4", "GAL80", "SWI5"]
    ax.set_xticklabels(labels=genes)
    ax.set_yticklabels(labels=genes)


def plot_roc_curve(roc_values_list: list, directed: bool):
    """Plot ROC curve of the respective calculated values of one or more models."""
    leg_labels = []
    plt.figure(figsize=(5, 5))
    if directed:
        ind_fpr = 2
        ind_tpr = 1
        title = "ROC curve for directed edges"
    else:
        ind_fpr = 4
        ind_tpr = 3
        title = "ROC curve for undirected edges"

    for roc_value_list in roc_values_list:
        plt.plot(roc_value_list[ind_fpr], roc_value_list[ind_tpr], "o-")
        leg_labels.append(roc_value_list[0])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(title)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(labels=leg_labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.show()
