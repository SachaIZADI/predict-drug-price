from dataclasses import dataclass
from pathlib import Path
from utils import get_git_root
from typing import List


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
    n_estimators = [50, 100, 150, 200]
    max_depth = [3, 5, 10, 15]
    learning_rate = [0.10, 0.15, 0.20, 0.25]
    reg_lambda = [0, 1, 1.25, 1.5, 1.75, 2]
    reg_alpha = [0, 1, 1.5, 2]


@dataclass
class Config:
    use_grid_search: bool = False
    use_cross_validation: bool = True
    visualize_results: bool = True
    save_model: bool = True

    model_path: Path = get_git_root() / "model" / "xgb.pkl"

    xgboost_params: XGBParams = XGBParams()
    grid_search_params: GridSearchParams = GridSearchParams()

    target: str = "price"
    features_to_use = [
        'label_plaquette', 'label_ampoule',
        'label_flacon', 'label_tube', 'label_stylo', 'label_seringue',
        'label_pilulier', 'label_sachet', 'label_comprime', 'label_gelule',
        'label_film', 'label_poche', 'label_capsule', 'count_plaquette',
        'count_ampoule', 'count_flacon', 'count_tube', 'count_stylo',
        'count_seringue', 'count_pilulier', 'count_sachet', 'count_comprime',
        'count_gelule', 'count_film', 'count_poche', 'count_capsule',
        'count_ml',
        'active_ingredient_feature_1', 'active_ingredient_feature_2',
        'active_ingredient_feature_3',
        'active_ingredient_feature_4', 'active_ingredient_feature_5',
        'active_ingredient_feature_6', 'active_ingredient_feature_7',
        'active_ingredient_feature_8', 'active_ingredient_feature_9',
        'active_ingredient_feature_10',
        "active_ingredients_count"
    ]


config = Config()
