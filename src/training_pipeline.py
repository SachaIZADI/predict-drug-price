from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from src.feature_engineering.feature_builder import FeatureBuilder
from src.model import model
from src.visualization import plot_feature_importance
from src.config import config


def train_model():

    # TODO :
    #  - [ ] Create 1 or 2 additional features
    #    - [x] Add `reimbursement_rate` + `marketing_declaration_date` + `marketing_authorization_date`
    #    - [ ] Create feature for 'pharmaceutical_companies'
    #  - [ ] Add Missing Value handling
    #  - [ ] Add debug module
    #    - [ ] Visualize quality of fit
    #    - [ ] Investigate unfit model
    #  - [ ] Finalize model pipeline
    #    - [x] Config
    #    - [ ] Training
    #  - [ ] Generate output
    #  - [ ] Update readme

    feature_builder = FeatureBuilder()
    feature_builder.fit()
    features_df = feature_builder.transform()

    # FIXME : to be updated
    features_df = features_df.dropna()

    X_train, y_train, *_, = FeatureBuilder.get_train_test_sets(
        features_df,
        features=config.features_to_use,
        target=config.target,
    )

    if config.use_grid_search:

        from dataclasses import asdict
        param_grid = asdict(config.grid_search_params)

        grid_search = GridSearchCV(model, param_grid, scoring='r2', refit=True, cv=3, verbose=10)
        grid_search.fit(X_train, y_train)

        print(grid_search.best_score_)
        print(grid_search.best_params_)

    if config.use_cross_validation:

        scores = cross_validate(
            model, X_train, y_train,
            cv=5,
            scoring='r2',
            return_train_score=True
        )

        print(
            scores['train_score'].mean(),
            scores['test_score'].mean()
        )

    model.fit(X_train, y_train)

    if config.visualize_results:
        plot_feature_importance(model)



    # /Users/izadisacha/Documents/MM_case/venv/bin/python /Users/izadisacha/Documents/MM_case/src/__main__.py
    # 0.6402693071000864 0.3240530179893231
    # 0.43582798689197394
    # {'alpha': 10, 'lambda': 10}

    # 0.9482052812895156 0.4671893256001427

    # 0.9461977645304589 0.5280953106977562
    #     max_depth=10,
    #     learning_rate=0.01,
    #     colsample_bytree=0.4,
    #     subsample=0.8,
    #     n_estimators=1000,
    #     reg_alpha=0.3,
    #     gamma=1,
    #     verbosity=1

    # With commercial 01/07
    # 0.9644703295278356 0.647770480569298

    # With comercial + pharma companies 01/07
    # 0.965119462085655 0.6468714741836734