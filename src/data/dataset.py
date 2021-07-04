import logging
from typing import Tuple, Sequence
from sklearn.model_selection import train_test_split as _train_test_split
import pandas as pd

from src.data.loader import load_data
from src.config import config


logger = logging.getLogger(__name__)


def get_dataset(train: bool = True, test: bool = True) -> pd.DataFrame:
    active_ingredients = load_data("active_ingredients.csv")
    drug_label_feature_eng = load_data("drug_label_feature_eng.csv")
    drugs_test = load_data("drugs_test.csv")
    drugs_train = load_data("drugs_train.csv")

    drug_label_feature_eng = drug_label_feature_eng.drop_duplicates(
        subset=["description"], keep="first"
    )
    active_ingredients = active_ingredients.groupby(
        ["drug_id"], as_index=False
    ).aggregate({"active_ingredient": list})

    dataset = pd.concat([drugs_train, drugs_test])
    dataset = dataset.merge(
        active_ingredients,
        on="drug_id",
        how="left",
        validate="1:1",
    )
    dataset = dataset.merge(
        drug_label_feature_eng,
        on="description",
        how="left",
        validate="m:1",
    )

    if not train:
        dataset = dataset.loc[~dataset["drug_id"].str.contains("train"), :]
    if not test:
        dataset = dataset.loc[~dataset["drug_id"].str.contains("test"), :]

    return dataset


def get_training_data(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, y = split_labels(dataset)
    X_train, _, y_train, _ = train_test_split(X, y)
    return X_train, y_train


def get_testing_data(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, y = split_labels(dataset)
    _, X_test, _, y_test = train_test_split(X, y)
    return X_test, y_test


def train_test_split(X: Sequence, y: Sequence) -> tuple:
    return _train_test_split(X, y, test_size=0.2, random_state=42)


def split_labels(
    df: pd.DataFrame, target: str = config.target
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target]
    X = df.drop([target], axis=1)
    return X, y
