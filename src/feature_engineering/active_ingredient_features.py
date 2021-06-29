import pandas as pd
from sklearn.decomposition import TruncatedSVD
from functools import cached_property

from src.feature_engineering.base_feature import Feature
from src.data_loader import DataLoader


class ActiveIngredientsFeature(Feature):

    SOURCE_FILES = ["active_ingredients"]
    N_COMPONENTS = 40

    def __init__(self):
        self.svd = TruncatedSVD(n_components=self.N_COMPONENTS)

    def fit(self):
        active_ingredients_df = self.active_ingredients_df
        active_ingredients_df = active_ingredients_df
        self.svd.fit(active_ingredients_df)

    def transform(self) -> pd.DataFrame:
        active_ingredients_df = self.active_ingredients_df
        active_ingredients_features_df = pd.DataFrame(
            self.svd.transform(active_ingredients_df),
            columns=[f"active_ingredient_feature_{i}" for i in range(1, self.N_COMPONENTS + 1)]
        )
        active_ingredients_df = pd.concat([
            active_ingredients_df.reset_index()[["drug_id"]], active_ingredients_features_df
        ], axis=1)
        return active_ingredients_df

    @cached_property
    def active_ingredients_df(self) -> pd.DataFrame:
        active_ingredients_df = DataLoader().load_data(self.SOURCE_FILES)["active_ingredients"]
        active_ingredients_df = (
            active_ingredients_df
            .assign(value=1)
            .pivot_table(index="drug_id", columns="active_ingredient", values="value")
            .fillna(0)
        )
        return active_ingredients_df
