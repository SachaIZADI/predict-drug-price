import pandas as pd
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def plot_feature_importance(
    model: Pipeline,
    importance_type: str = "weight",
    figsize: Tuple[int, int] = (20, 8),
):

    feature_names = model[0].get_feature_names()
    index_renaming = {f"f{i}": feature_names[i] for i in range(len(feature_names))}

    feature_importance = (
        model[1].regressor_.get_booster().get_score(importance_type=importance_type)
    )
    feature_importance = (
        pd.DataFrame(
            data=feature_importance.values(),
            index=feature_importance.keys(),
            columns=["score"],
        )
        .rename(index=index_renaming)
        .sort_values(by="score", ascending=False)
    )

    feature_importance.plot(kind="barh", figsize=figsize)
    plt.show()


def plot_errors(
    y_true: np.array,
    y_pred: np.array,
    log: bool = False,
    figsize: Tuple[int, int] = (20, 20),
):
    plt.figure(figsize=figsize)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")

    if log:
        plt.xscale("log")
        plt.yscale("log")

    plt.plot(y_true, y_pred, "o")
    plt.plot([0, y_true.max()], [0, y_true.max()], "-")
    plt.show()
