from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from dataclasses import asdict

from src.feature_engineering.feature_generator import features_generator
from src.config import config

regressor = XGBRegressor(**asdict(config.xgboost_params))

model = Pipeline([
    ('features', features_generator),
    ('estimator', regressor)
])

