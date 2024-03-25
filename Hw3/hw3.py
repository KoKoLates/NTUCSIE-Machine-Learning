
import numpy as np

from abc import ABC, abstractmethod

class Regressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self) -> None:
        pass


class LinearRegressor(object):
    ...


class LogisticRegressor(object):
    ...
