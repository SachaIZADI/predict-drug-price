from xgboost import XGBRegressor
from dataclasses import asdict

from src.config import config

Model = XGBRegressor
model = Model(**asdict(config.xgboost_params))
