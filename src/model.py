from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from dataclasses import asdict
import numpy as np

from src.feature_engineering.feature_generator import features_generator
from src.config import config

regressor = XGBRegressor(**asdict(config.xgboost_params))

model = Pipeline([
    ('features', features_generator),
    ('estimator', regressor)
])


class TargetLogTransformation:

    @classmethod
    def transform(cls, y: np.array) -> np.array:
        return np.log(y)

    @classmethod
    def un_transform(cls, y: np.array) -> np.array:
        return np.exp(y)

