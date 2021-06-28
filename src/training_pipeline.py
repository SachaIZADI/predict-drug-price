from sklearn.model_selection import cross_validate

from src.feature_engineering.feature_builder import FeatureBuilder
from src.model import model


def train_model():

    FEATURES = [
        'label_plaquette', 'label_ampoule',
        'label_flacon', 'label_tube', 'label_stylo', 'label_seringue',
        'label_pilulier', 'label_sachet', 'label_comprime', 'label_gelule',
        'label_film', 'label_poche', 'label_capsule', 'count_plaquette',
        'count_ampoule', 'count_flacon', 'count_tube', 'count_stylo',
        'count_seringue', 'count_pilulier', 'count_sachet', 'count_comprime',
        'count_gelule', 'count_film', 'count_poche', 'count_capsule',
        'count_ml'
    ]

    feature_builder = FeatureBuilder()
    features_df = feature_builder.transform()

    # FIXME : to be updated
    features_df = features_df.dropna()

    X_train, y_train, X_test, y_test = FeatureBuilder.get_train_test_sets(
        features_df,
        features=FEATURES,
        target="price"
    )

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
