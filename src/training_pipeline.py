from src.feature_engineering.feature_builder import FeatureBuilder
from src.model import Model
from src.metrics import ValidationMetrics


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

    model = Model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    metrics = ValidationMetrics(y_pred=y_pred, y_true=y_train)

    print(metrics.mape)
    print(metrics.r2)