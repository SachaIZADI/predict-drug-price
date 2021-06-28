import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error


class ValidationMetrics:

    def __init__(self, y_true: pd.Series, y_pred: pd.Series):
        self.y_true = y_true
        self.y_pred = y_pred

    @property
    def r2(self) -> float:
        return r2_score(self.y_true, self.y_pred)

    @property
    def mape(self) -> float:
        return mean_absolute_percentage_error(self.y_true, self.y_pred)