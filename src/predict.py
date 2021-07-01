from logging import getLogger

import pandas as pd
import pickle

import src.data.dataset as ds
from src.data.loader import save_data
from src.model import TargetLogTransformation
from src.config import config


logger = getLogger(__name__)


def predict():

    data = ds.get_dataset(train=False, test=True)
    X, _ = ds.split_labels(data)

    logger.info(f'Loading model…')
    with open(config.model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f'Predicting on test dataset…')
    y_pred = model.predict(X)
    y_pred = TargetLogTransformation.transform(y_pred)

    y_pred = pd.DataFrame(y_pred, columns=["price"])

    results = pd.concat([X[["drug_id"]].reset_index(drop=True), y_pred.reset_index(drop=True)], axis=1)

    save_data(results, "submission.csv")