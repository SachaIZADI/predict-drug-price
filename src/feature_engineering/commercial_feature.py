import pandas as pd
import numpy as np
from functools import cached_property
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering.base_feature import Feature
from src.data_loader import DataLoader


class CommercialFeature(Feature):

    SOURCE_FILES = ["drugs_train", "drugs_test"]
    FEATURES_TO_ENCODE = [
        "administrative_status",
        'marketing_status',
        'approved_for_hospital_use',
        'marketing_authorization_process',
    ]
    N_COMPONENTS = 10

    def __init__(self):
        self.commercial_features_df = pd.concat([*DataLoader().load_data(self.SOURCE_FILES).values()])
        self.one_hot_encoder = OneHotEncoder()

    def fit(self):
        self.one_hot_encoder.fit(
            self.commercial_features_df[self.FEATURES_TO_ENCODE]
        )

    def transform(self) -> pd.DataFrame:

        commercial_features_df = self.commercial_features_df.copy()
        commercial_features_df["reimbursement_rate"] = (
            commercial_features_df["reimbursement_rate"].str.replace("%", "").astype(int) / 100
        )
        commercial_features_df['marketing_authorization_date'] //= 10000
        commercial_features_df['marketing_declaration_date'] //= 10000

        one_hot_data = self.one_hot_encoder.transform(commercial_features_df[self.FEATURES_TO_ENCODE])
        one_hot_features = list(np.concatenate(self.one_hot_encoder.categories_))
        one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_data, columns=one_hot_features)

        commercial_features_df = pd.concat([
            commercial_features_df.reset_index(drop=True),
            one_hot_df.reset_index(drop=True)
        ], axis=1)


        return commercial_features_df[[
            "drug_id",
            "reimbursement_rate",
            'marketing_authorization_date',
            'marketing_declaration_date',
            *one_hot_features,
        ]]