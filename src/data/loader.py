import pandas as pd
from src.utils import get_git_root


def load_data(file_name: str) -> pd.DataFrame:
    path_to_file = get_git_root() / "data" / file_name
    return pd.read_csv(path_to_file, sep=",")


def save_data(df: pd.DataFrame, file_name: str):
    path_to_file = get_git_root() / "data" / file_name
    df.to_csv(path_to_file, sep=",", index=False)