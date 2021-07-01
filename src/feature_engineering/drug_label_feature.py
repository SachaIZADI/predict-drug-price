import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering.base_feature import BaseFeature, ColumnExtractorMixin


class LabelPlaquette(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_plaquette'


class LabelAmpoule(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_ampoule'


class LabelFlacon(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_flacon'


class LabelTube(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_tube'


class LabelStylo(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_stylo'


class LabelSeringue(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_seringue'


class LabelPillulier(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_pilulier'


class LabelSachet(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_sachet'


class LabelComprime(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_comprime'


class LabelGelule(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_gelule'


class LabelFilm(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_film'


class LabelPoche(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_poche'


class LabelCapsule(BaseFeature, ColumnExtractorMixin):
    _cname = 'label_capsule'


class CountPlaquette(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_plaquette'


class CountAmpoule(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_ampoule'


class CountFlacon(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_flacon'


class CountTube(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_tube'


class CountStylo(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_stylo'


class CountSeringue(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_seringue'


class CountPillulier(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_pilulier'


class CountSachet(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_sachet'


class CountComprime(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_comprime'


class CountGelule(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_gelule'


class CountFilm(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_film'


class CountPoche(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_poche'


class CountCapsule(BaseFeature, ColumnExtractorMixin):
    _cname = 'count_capsule'
