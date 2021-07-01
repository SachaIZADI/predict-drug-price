import pandas as pd

from src.feature_engineering.base_feature import Feature
from src.data_loader import DataLoader

class DrugLabelFeature(Feature):

    SOURCE_FILES = ["drugs_test", "drugs_train", "drug_label_feature_eng"]

    def __init__(self):
        self.input_data = DataLoader().load_data(self.SOURCE_FILES)

    def fit(self):
        pass

    def transform(self) -> pd.DataFrame:
        drugs_df = pd.concat([self.input_data["drugs_test"], self.input_data["drugs_train"]])
        drugs_df = drugs_df[["drug_id", "description"]]
        main_features_df = self.input_data["drug_label_feature_eng"]
        main_features_df = main_features_df.drop_duplicates(subset=["description"], keep="first")

        main_feature_df = drugs_df.merge(
           main_features_df,
            on=["description"],
            how="left",
            validate="m:1"
        )
        main_feature_df = main_feature_df.drop(columns=["description"])

        return main_feature_df
