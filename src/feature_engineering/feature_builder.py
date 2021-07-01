import pandas as pd
from typing import Tuple, List

from src.utils import reduce_data_frames
from src.feature_engineering.drug_label_feature import DrugLabelFeature
from src.feature_engineering.active_ingredient_feature import ActiveIngredientsFeature
from src.feature_engineering.commercial_feature import CommercialFeature
from src.data_loader import DataLoader


class FeatureBuilder:

    def __init__(self):
        self._features = [
            DrugLabelFeature(),
            ActiveIngredientsFeature(),
            CommercialFeature(),
        ]

    @property
    def index(self) -> pd.DataFrame:
        SOURCE_FILES = ["drugs_test", "drugs_train"]
        input_data = DataLoader().load_data(SOURCE_FILES)
        index = pd.concat([
            input_data["drugs_test"].assign(label="test"),
            input_data["drugs_train"].assign(label="train"),
        ])
        index = index[["drug_id", "label", "price"]].drop_duplicates(subset=["drug_id"], keep="first")
        return index

    def fit(self):
        for feature in self._features:
            feature.fit()

    def transform(self) -> pd.DataFrame:
        feature_dfs = [
            feature.transform() for feature in self._features
        ]
        return reduce_data_frames([self.index, *feature_dfs], on=["drug_id"])

    @classmethod
    def get_train_test_sets(
        cls,
        features_df: pd.DataFrame,
        features: List[str],
        target: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

        return (
            features_df.loc[features_df["label"] == "train", features],
            features_df.loc[features_df["label"] == "train", target],
            features_df.loc[features_df["label"] == "test", features],
            features_df.loc[features_df["label"] == "test", target],
        )
