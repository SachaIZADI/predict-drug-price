from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

Model = XGBRegressor
model = Model(
    max_depth=10,
    learning_rate=0.01,
    colsample_bytree=0.4,
    subsample=0.8,
    n_estimators=1000,
    reg_alpha=0.3,
    gamma=10,
    verbosity=1
)
