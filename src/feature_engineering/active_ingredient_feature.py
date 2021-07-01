import pandas as pd
from sklearn.decomposition import TruncatedSVD
from functools import cached_property

from sklearn.preprocessing import OneHotEncoder
from src.feature_engineering.base_feature import BaseFeature


class ActiveIngredientsCount(BaseFeature):
    def transform(self, X):
        return pd.DataFrame(X["active_ingredient"].apply(len))

class ActiveIngredientsFeature(BaseFeature):

    # TODO: optimize this hyperparameter
    N_COMPONENTS = 10

    def fit(self, X, y=None):
        X = self.prepare_encoder_input(X)
        encoder = OneHotEncoder(sparse=False)
        self.encoder = encoder.fit(X[["active_ingredient"]])
        X_encoded = self.encoder.transform(X[["active_ingredient"]])

        X = self.prepare_svd_input(X, X_encoded)
        self.svd = TruncatedSVD(n_components=self.N_COMPONENTS)
        self.svd.fit(X)
        return self

    def transform(self, X):
        X = self.prepare_encoder_input(X)
        X_encoded = self.encoder.transform(X[["active_ingredient"]])
        X = self.prepare_svd_input(X, X_encoded)
        return self.svd.transform(X)

    @staticmethod
    def prepare_encoder_input(X_input) -> pd.DataFrame:
        X = X_input[["drug_id", "active_ingredient"]].explode("active_ingredient").reset_index(drop=True)
        return X

    @staticmethod
    def prepare_svd_input(X_input, X_encoded) -> pd.DataFrame:
        X = pd.concat([
            X_input[["drug_id"]],
            pd.DataFrame(X_encoded),
        ], axis=1)
        return X.groupby(["drug_id"], as_index=True).sum()
