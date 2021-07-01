import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.feature_engineering.base_feature import BaseFeature


class AdministrativeStatus(BaseFeature):

    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False)
        self.encoder = encoder.fit(X[["administrative_status"]])
        return self

    def transform(self, X):
        return self.encoder.transform(X[["administrative_status"]])


class MarketingStatus(BaseFeature):

    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False)
        self.encoder = encoder.fit(X[['marketing_status']])
        return self

    def transform(self, X):
        return self.encoder.transform(X[['marketing_status']])


class ApprovedForHospitalUse(BaseFeature):

    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False)
        self.encoder = encoder.fit(X[['approved_for_hospital_use']])
        return self

    def transform(self, X):
        return self.encoder.transform(X[['approved_for_hospital_use']])


class MarketingAuthorizationProcess(BaseFeature):

    def fit(self, X, y=None):
        encoder = OneHotEncoder(sparse=False)
        self.encoder = encoder.fit(X[['marketing_authorization_process']])
        return self

    def transform(self, X):
        return self.encoder.transform(X[['marketing_authorization_process']])


class ReimbursementRate(BaseFeature):
    def transform(self, X):
        return pd.DataFrame(X["reimbursement_rate"].str.replace("%", "").astype(int) / 100)


class MarketingAuthorizationDate(BaseFeature):
    def transform(self, X):
        return X[['marketing_authorization_date']] // 10000


class MarketingDeclarationDate(BaseFeature):
    def transform(self, X):
        return X[['marketing_declaration_date']] // 10000
