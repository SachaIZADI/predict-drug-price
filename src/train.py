from logging import getLogger

from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_log_error

import pickle
from dataclasses import asdict
import numpy as np

from src.model import model
import src.data.dataset as ds
from src.visualization import plot_feature_importance, plot_errors
from src.config import config


logger = getLogger(__name__)


def train():

    logger.info(f"Training {model.__class__.__name__} on dataset…")

    data = ds.get_dataset(train=True, test=False)
    X, y = ds.split_labels(data)

    X_train, X_test, y_train, y_test = ds.train_test_split(X, y)

    # --- Display score for Baseline model (just predicting the mean)
    y_pred = np.ones(y.shape) * y.mean()
    logger.info(
        f"Baseline 'dummy' model `mean_squared_log_error`: {mean_squared_log_error(y, y_pred)}"
    )

    if config.use_grid_search:
        logger.info(f"Launching grid search on {model.__class__.__name__}…")
        param_grid = asdict(config.grid_search_params)

        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring="neg_mean_squared_log_error",
            refit=True,
            cv=3,
            verbose=3,
        )
        grid_search.fit(X_train, y_train)

        # TODO : how to pass the best params to the model automatically ?
        logger.info(f"Grid search best score: {grid_search.best_score_}")
        logger.info(f"Grid search best score: {grid_search.best_params_}")

    if config.use_cross_validation:
        logger.info(f"Launching cross validation on {model.__class__.__name__}…")
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=config.cv_k_fold,
            scoring="neg_mean_squared_log_error",
            return_train_score=True,
        )
        logger.info(
            f"Mean of ‘neg_mean_squared_log_error’ on TRAIN set: {scores['train_score'].mean()}"
        )
        logger.info(
            f"Variance of ‘neg_mean_squared_log_error’ on TRAIN set: {scores['train_score'].var()}"
        )

        logger.info(
            f"Mean of ‘neg_mean_squared_log_error’ on TEST set: {scores['test_score'].mean()}"
        )
        logger.info(
            f"Variance of ‘neg_mean_squared_log_error’ on TEST set: {scores['test_score'].var()}"
        )

    logger.info(f"Training of {model.__class__.__name__} for train/test score")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info(
        f"‘mean_squared_log_error’ on TEST set: {mean_squared_log_error(y_true=y_test, y_pred=y_pred)}"
    )

    if config.visualize_results:
        logger.info(
            f"Training of {model.__class__.__name__} for visualization & debug…"
        )
        model.fit(X_train, y_train)

        logger.info(f"Launching visualization…")
        plot_feature_importance(model)
        plot_errors(y_pred=y_pred, y_true=y_test, log=True)

    logger.info(f"Final training of {model.__class__.__name__} on fulldataset…")
    model.fit(X, y)

    if config.save_model:
        with open(config.model_path, "wb+") as f:
            pickle.dump(model, f)
