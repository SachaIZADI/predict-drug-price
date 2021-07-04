import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.feature_engineering.base_feature import BaseFeature


class AdministrativeStatus(BaseFeature):
    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder = encoder.fit(X[["administrative_status"]])
        return self

    def transform(self, X):
        return self.encoder.transform(X[["administrative_status"]])

    def get_feature_names(self):
        return self.encoder.get_feature_names()


class MarketingStatus(BaseFeature):
    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder = encoder.fit(X[["marketing_status"]])
        return self

    def transform(self, X):
        return self.encoder.transform(X[["marketing_status"]])

    def get_feature_names(self):
        return self.encoder.get_feature_names()


class ApprovedForHospitalUse(BaseFeature):
    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder = encoder.fit(X[["approved_for_hospital_use"]])
        return self

    def transform(self, X):
        return self.encoder.transform(X[["approved_for_hospital_use"]])

    def get_feature_names(self):
        return self.encoder.get_feature_names()


class MarketingAuthorizationProcess(BaseFeature):
    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder = encoder.fit(X[["marketing_authorization_process"]])
        return self

    def transform(self, X):
        return self.encoder.transform(X[["marketing_authorization_process"]])

    def get_feature_names(self):
        return self.encoder.get_feature_names()


class ReimbursementRate(BaseFeature):
    def transform(self, X):
        return pd.DataFrame(
            X["reimbursement_rate"].str.replace("%", "").astype(int) / 100
        )


class MarketingAuthorizationDate(BaseFeature):
    def transform(self, X):
        return X[["marketing_authorization_date"]] // 10000


class MarketingDeclarationDate(BaseFeature):
    def transform(self, X):
        return X[["marketing_declaration_date"]] // 10000
