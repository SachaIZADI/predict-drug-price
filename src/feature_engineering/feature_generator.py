from sklearn.pipeline import FeatureUnion

from .active_ingredient_feature import *
from .commercial_feature import *
from .drug_label_feature import *

FEATURES_LIST = [
    # ---- Active Ingredients features -----
    ActiveIngredientsCount,
    ActiveIngredientsFeature,
    # ---- Commercial features -----
    AdministrativeStatus,
    MarketingStatus,
    ApprovedForHospitalUse,
    MarketingAuthorizationProcess,
    ReimbursementRate,
    MarketingAuthorizationDate,
    MarketingDeclarationDate,
    # ---- Drug labels features -----
    LabelPlaquette,
    LabelAmpoule,
    LabelFlacon,
    LabelTube,
    LabelStylo,
    LabelSeringue,
    LabelPillulier,
    LabelSachet,
    LabelComprime,
    LabelGelule,
    LabelFilm,
    LabelPoche,
    LabelCapsule,
    CountPlaquette,
    CountAmpoule,
    CountFlacon,
    CountTube,
    CountStylo,
    CountSeringue,
    CountPillulier,
    CountSachet,
    CountComprime,
    CountGelule,
    CountFilm,
    CountPoche,
    CountCapsule,
]

FEATURES_STORE = [(f.name(), f()) for f in FEATURES_LIST]

features_generator = FeatureUnion(FEATURES_STORE)
