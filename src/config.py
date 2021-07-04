from dataclasses import dataclass
from pathlib import Path
from utils import get_git_root


@dataclass
class XGBParams:
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 500


@dataclass
class GridSearchParams:
    estimator__regressor__n_estimators: tuple = (50, 100, 500, 1000)
    estimator__regressor__max_depth: tuple = (5, 10, 15)
    estimator__regressor__learning_rate: tuple = (0.05, 0.10, 0.15)


@dataclass
class Config:
    use_grid_search: bool = False
    use_cross_validation: bool = True
    cv_k_fold: int = 5
    visualize_results: bool = True
    save_model: bool = True

    model_path: Path = get_git_root() / "model" / "pipeline.pkl"

    n_components_active_ingrendients_svd: int = 10
    xgboost_params: XGBParams = XGBParams()
    grid_search_params: GridSearchParams = GridSearchParams()

    target: str = "price"


config = Config()
