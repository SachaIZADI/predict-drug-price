import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def plot_feature_importance(
    model: XGBRegressor,
    importance_type: str = "weight",
    figsize: Tuple[int, int] = (20, 8),
):
    feature_importance = model.get_booster().get_score(importance_type=importance_type)

    feature_importance = pd.DataFrame(
        data=feature_importance.values(),
        index=feature_importance.keys(),
        columns=["score"]
    ).sort_values(by="score", ascending=False)

    feature_importance.plot(kind='barh', figsize=figsize)
    plt.show()
