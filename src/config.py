from dataclasses import dataclass
from pathlib import Path
from utils import get_git_root
from typing import List, Tuple


@dataclass
class XGBParams:
    max_depth: int = 10
    learning_rate: float = 0.01
    colsample_bytree: float = 0.4
    subsample: float = 0.8
    n_estimators: int = 1000
    reg_alpha: float = 0.3
    gamma: float = 10


@dataclass
class GridSearchParams:
    estimator__n_estimators: tuple = (50, 100, 150, 200)
    estimator__max_depth: tuple = (5, 10, 15)
    estimator__learning_rate: tuple = (0.10, 0.15, 0.20)
    estimator__reg_lambda: tuple = (0, 1, 1.5, 2)
    estimator__reg_alpha: tuple = (0, 1, 1.5, 2)


@dataclass
class Config:
    use_grid_search: bool = False
    use_cross_validation: bool = False
    cv_k_fold: int = 5
    visualize_results: bool = True
    save_model: bool = True

    model_path: Path = get_git_root() / "model" / "pipeline.pkl"

    n_components_active_ingrendients_svd: int = 10
    xgboost_params: XGBParams = XGBParams()
    grid_search_params: GridSearchParams = GridSearchParams()

    target: str = "price"

config = Config()
