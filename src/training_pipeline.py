from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from src.feature_engineering.feature_builder import FeatureBuilder
from src.model import model


def train_model(
    use_grid_search: bool = True
):

    # TODO :
    #  - [ ] Create 1 or 2 additional features
    #  - [ ] Add Missing Value handling
    #  - [ ] Finalize model pipeline


    FEATURES = [
        'label_plaquette', 'label_ampoule',
        'label_flacon', 'label_tube', 'label_stylo', 'label_seringue',
        'label_pilulier', 'label_sachet', 'label_comprime', 'label_gelule',
        'label_film', 'label_poche', 'label_capsule', 'count_plaquette',
        'count_ampoule', 'count_flacon', 'count_tube', 'count_stylo',
        'count_seringue', 'count_pilulier', 'count_sachet', 'count_comprime',
        'count_gelule', 'count_film', 'count_poche', 'count_capsule',
        'count_ml',
        'active_ingredient_feature_1', 'active_ingredient_feature_2',
        'active_ingredient_feature_3',
        'active_ingredient_feature_4', 'active_ingredient_feature_5',
        'active_ingredient_feature_6', 'active_ingredient_feature_7',
        'active_ingredient_feature_8', 'active_ingredient_feature_9',
        'active_ingredient_feature_10',
        "active_ingredients_count"
    ]

    feature_builder = FeatureBuilder()
    feature_builder.fit()
    features_df = feature_builder.transform()

    # FIXME : to be updated
    features_df = features_df.dropna()

    X_train, y_train, *_, = FeatureBuilder.get_train_test_sets(
        features_df,
        features=FEATURES,
        target="price"
    )

    if use_grid_search:

        param_grid = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 5, 10, 15],
            "learning_rate": [0.10, 0.15, 0.20, 0.25],
            "lambda": [0, 1, 1.25, 1.5, 1.75, 2],
        }
        grid_search = GridSearchCV(model, param_grid, scoring='r2', refit=True, cv=5)
        grid_search.fit(X_train, y_train)

        print(grid_search.best_score_)
        print(grid_search.best_params_)


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

    # model.fit(X_train, y_train)

    # /Users/izadisacha/Documents/MM_case/venv/bin/python /Users/izadisacha/Documents/MM_case/src/__main__.py
    # 0.6402693071000864 0.3240530179893231
    # 0.3636898504750632
    # {'learning_rate': 0.15, 'max_depth': 5, 'n_estimators': 100}

    # 0.9482052812895156 0.4671893256001427
