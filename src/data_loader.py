import pandas as pd
from typing import Dict, List
from src.utils import get_git_root


class DataLoader:

    def __init__(self):
        self._root_path = get_git_root() / "data"

    def load_data(self, files_to_load: List[str]) -> Dict[str, pd.DataFrame]:
        return {
            file_name: pd.read_csv(self._root_path / f"{file_name}.csv", sep=",")
            for file_name in files_to_load
        }