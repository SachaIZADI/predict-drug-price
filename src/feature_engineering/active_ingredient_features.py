import pandas as pd
from sklearn.decomposition import TruncatedSVD
from functools import cached_property

from src.feature_engineering.base_feature import Feature
from src.data_loader import DataLoader


class ActiveIngredientsFeature(Feature):

    SOURCE_FILES = ["active_ingredients"]
    N_COMPONENTS = 10

    def __init__(self):
        self.active_ingredients_df = DataLoader().load_data(self.SOURCE_FILES)["active_ingredients"]
        self.svd = TruncatedSVD(n_components=self.N_COMPONENTS)

    def fit(self):
        active_ingredients_df = self.active_ingredients_one_hot_encoding_df
        active_ingredients_df = active_ingredients_df
        self.svd.fit(active_ingredients_df)

    def transform(self) -> pd.DataFrame:
        active_ingredients_df = self.active_ingredients_one_hot_encoding_df
        active_ingredients_features_df = pd.DataFrame(
            self.svd.transform(active_ingredients_df),
            columns=[f"active_ingredient_feature_{i}" for i in range(1, self.N_COMPONENTS + 1)]
        )
        active_ingredients_df = pd.concat([
            active_ingredients_df.reset_index()[["drug_id"]], active_ingredients_features_df
        ], axis=1)

        active_ingredients_df = active_ingredients_df.merge(
            self.active_ingredients_count_df,
            on=["drug_id"],
            how="left",
            validate="1:1",
        )

        return active_ingredients_df

    @cached_property
    def active_ingredients_one_hot_encoding_df(self) -> pd.DataFrame:
        active_ingredients_df = (
            self.active_ingredients_df
            .assign(value=1)
            .pivot_table(index="drug_id", columns="active_ingredient", values="value")
            .fillna(0)
        )
        return active_ingredients_df

    @cached_property
    def active_ingredients_count_df(self) -> pd.DataFrame:
        active_ingredients_count_df = (
            self.active_ingredients_df
            .groupby(["drug_id"], as_index=False)
            .count()
            .rename(columns={"active_ingredient": "active_ingredients_count"})
        )
        return active_ingredients_count_df
