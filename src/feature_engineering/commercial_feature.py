import pandas as pd
import numpy as np
from functools import cached_property
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD

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
        self.one_hot_encoder_pharma_companies = OneHotEncoder()
        self.svd = TruncatedSVD(n_components=self.N_COMPONENTS)

    def fit(self):
        self.one_hot_encoder.fit(
            self.commercial_features_df[self.FEATURES_TO_ENCODE]
        )

        one_hot_pharma_companies = self.one_hot_encoder_pharma_companies.fit_transform(
            self.commercial_features_df[["pharmaceutical_companies"]]
        )
        self.svd.fit(one_hot_pharma_companies)


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

        one_hot_pharma_companies = self.one_hot_encoder_pharma_companies.transform(
            self.commercial_features_df[["pharmaceutical_companies"]]
        )
        pharma_companies_features = pd.DataFrame(
            self.svd.transform(one_hot_pharma_companies),
            columns=[f"pharma_companies_{i}" for i in range(1, self.N_COMPONENTS + 1)]
        )

        commercial_features_df = pd.concat([
            commercial_features_df.reset_index(drop=True),
            one_hot_df.reset_index(drop=True),
            pharma_companies_features.reset_index(drop=True),
        ], axis=1)

        return commercial_features_df[[
            "drug_id",
            "reimbursement_rate",
            'marketing_authorization_date',
            'marketing_declaration_date',
            *one_hot_features,
            *[f"pharma_companies_{i}" for i in range(1, self.N_COMPONENTS + 1)],
        ]]