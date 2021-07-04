import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering.base_feature import BaseFeature
from src.config import config


class ActiveIngredientsCount(BaseFeature):
    def transform(self, X):
        return pd.DataFrame(X["active_ingredient"].apply(len))


class ActiveIngredientsFeature(BaseFeature):
    def fit(self, X, y=None):
        X = self.prepare_encoder_input(X)
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder = encoder.fit(X[["active_ingredient"]])
        X_encoded = self.encoder.transform(X[["active_ingredient"]])

        X = self.prepare_svd_input(X, X_encoded)
        self.svd = TruncatedSVD(
            n_components=config.n_components_active_ingrendients_svd
        )
        self.svd.fit(X)
        return self

    def transform(self, X):
        X = self.prepare_encoder_input(X)
        X_encoded = self.encoder.transform(X[["active_ingredient"]])
        X = self.prepare_svd_input(X, X_encoded)
        return pd.DataFrame(
            self.svd.transform(X),
            columns=[
                f"active_ingredient_feature_{i}"
                for i in range(1, config.n_components_active_ingrendients_svd + 1)
            ],
        )

    def get_feature_names(self):
        return [
            f"active_ingredient_feature_{i}"
            for i in range(1, config.n_components_active_ingrendients_svd + 1)
        ]

    @staticmethod
    def prepare_encoder_input(X_input) -> pd.DataFrame:
        X = (
            X_input[["drug_id", "active_ingredient"]]
            .explode("active_ingredient")
            .reset_index(drop=True)
        )
        return X

    @staticmethod
    def prepare_svd_input(X_input, X_encoded) -> pd.DataFrame:
        X = pd.concat(
            [
                X_input[["drug_id"]],
                pd.DataFrame(X_encoded),
            ],
            axis=1,
        )
        return X.groupby(["drug_id"], as_index=True).sum()
