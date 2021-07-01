import git
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing import List, Union
from functools import reduce
import pandas as pd


def get_git_root() -> Path:
    git_repo = git.Repo("", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


def reduce_data_frames(
    frames: List[pd.DataFrame], on: List[str], how: str = "left"
) -> pd.DataFrame:

    def single_merge(left_df: pd.DataFrame, right_df: pd.DataFrame):
        return left_df.merge(right_df, on=on, how=how)

    return reduce(single_merge, frames)


SklearnEstimator = Union[Pipeline, BaseEstimator]