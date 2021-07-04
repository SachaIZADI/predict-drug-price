import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from dataclasses import asdict

from src.feature_engineering.feature_generator import features_generator
from src.config import config

regressor = TransformedTargetRegressor(
    regressor=XGBRegressor(**asdict(config.xgboost_params)),
    func=np.log,
    inverse_func=np.exp,
)

model = Pipeline([
    ('features', features_generator),
    ('estimator', regressor)
])