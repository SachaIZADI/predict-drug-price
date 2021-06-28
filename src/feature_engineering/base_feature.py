from abc import ABC

import pandas as pd


class Feature(ABC):

    def __init__(self, **kwargs):
        pass

    def fit(self, **kwargs):
        pass

    def transform(self, **kwargs) -> pd.DataFrame:
        pass