import pandas as pd
from typing import Dict
from src.utils import get_git_root


class DataLoader:

    def __init__(self):
        self._root_path = get_git_root() / "data"
        self.__files_to_load = [
            "active_ingredients",
            "drug_label_feature_eng",
            "drugs_test",
            "drugs_train",
        ]

    def load_data(self) -> Dict[str, pd.DataFrame]:
        return {
            file_name: pd.read_csv(self._root_path / f"{file_name}.csv", sep=",")
            for file_name in self.__files_to_load
        }

    def consolidate_data(self, loaded_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        input_df = pd.concat([loaded_data["drugs_train"], loaded_data["drugs_test"]])

        features_df = loaded_data["drug_label_feature_eng"].drop_duplicates(subset="description", keep="first")

        active_ingredients_df = loaded_data["active_ingredients"]
        active_ingredients_df = active_ingredients_df.groupby(["drug_id"], as_index=False).aggregate(list)

        input_df = input_df.merge(
            features_df,
            on="description",
            how="left",
            validate="m:1",
        )

        input_df = input_df.merge(
            active_ingredients_df,
            on="drug_id",
            how="left",
            validate="m:1",
        )
        return input_df

    def create_input_data(self) -> pd.DataFrame:
        loaded_data = self.load_data()
        consolidated_data = self.consolidate_data(loaded_data)
        return consolidated_data