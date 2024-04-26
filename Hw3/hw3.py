# -*- coding: utf-8 -*-

import random
import numpy as np

from tqdm import tqdm
from typing import Callable

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Configure(object):

    loss: Callable              # loss function
    epoch: int = 1000           # example epoches
    eta: float = 0.001          # learning rate
    stop: float | None = None   # iteration stop
    w: np.ndarray = None        # initial weights

    @staticmethod
    def load_data(
        file_path: str, Q: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """ load in the dataset from files 
        @param file_path: the dataset file path
        @param bias: add bias to dataset or not (with no value)
        @return: `(x, y)` respectively in tuple
        """
        with open(file_path, "r") as file:
            data: list[list[float]] = []
            for line in file.readlines():
                data.append([float(value) for value in line.split()])

        x: np.ndarray = np.array([v[:-1] for v in data])
        y: np.ndarray = np.array([v[-1]  for v in data])

        if Q is not None:
            x = Configure.transform(x, Q)

        ## adding the bias to the weight
        x = np.insert(x, 0, 1, axis=1)

        return x, y
    
    @staticmethod
    def transform(x: np.ndarray, Q: int) -> np.ndarray:
        """ homogeneous Q-th order polynomial transform 
        @param x: input data in x-space
        @param Q: the order of polynomial to be transform
        """
        new_x: list = []
        for k in range(2, Q + 1):
            new_x.append([[pow(j, k) for j in i] for i in x])
    
        return np.hstack((x, *np.array(new_x)))


class Loss(object):
    """ a static class used to store the loss function 
    and each error measurements """
    @staticmethod
    def mse(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> np.float32:
        """ average mean square error """
        s: np.ndarray = np.square((x @ w) - y)
        return s.mean()

    @staticmethod
    def ce(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> np.float32:
        """ average cross-entropy error """
        s: np.ndarray = np.log(1 + np.exp(-y * (x @ w)))
        return s.mean()

    @staticmethod
    def one(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> np.float32:
        """ average 1/0 (true/false count) error """
        def sign(x: np.ndarray) -> np.ndarray:
            return np.where(x < 0, 1, 0)

        return sign((x @ w) * y).mean()


class Regressor(ABC):
    """ An abstract class for derived as each type of regressor """
    def __init__(
        self, x: np.ndarray, y: np.ndarray, cfg: Configure
    ) -> None:
        """ constructor of regressor
        @param x: input data
        @param y: labeled data
        @param cfg: configuration with learning rate, loss function and etc.
        """
        super().__init__()
        self.x, self.y = x, y
        self.cfg: Configure = cfg

        self.w: np.ndarray = None if \
            self.cfg.w is None else self.cfg.w

    @staticmethod
    def select(x: np.ndarray, y: np.ndarray) -> tuple:
        """ random sample a data with label from dataset 
        @param x: input dataset
        @param y: labeled data of dataset 
        @return: sampled data
        """
        index: int = random.randint(0, len(x) - 1)
        return x[index], y[index]
    
    @staticmethod
    def logis(value: np.ndarray) -> np.ndarray:
        """ logistic function """
        return 1 / (1 + np.exp(-value))

    @abstractmethod
    def fit(self) -> None: ...


class Linear(Regressor):
    """ linear regression """
    def fit(self) -> tuple:
        """ @return: tuple of `(weight, in-sample error)` """
        if self.w is None:
            self.w: np.ndarray = np.linalg.pinv(self.x) @ self.y

        self.error: np.float32 = self.cfg.loss(self.x, self.y, self.w)
        return self.w, self.error


class Linear_SGD(Regressor):
    """ linear regression with stochastic gradient descent """
    def fit(self) -> int:
        """ @return: iteration times """
        if self.w is None:
            self.w: np.ndarray = np.zeros(len(self.x[0]))

        error: float = float("inf")
        update_times: int = 0
        while error > self.cfg.stop:
            x, y = Regressor.select(self.x, self.y)
            self.w += self.cfg.eta * 2 * (y - (self.w @ x)) * x
            error = self.cfg.loss(self.x, self.y, self.w)
            update_times += 1

        return update_times


class Logistic_SGD(Regressor):
    """ logistic regression with stochastic gradient descent """
    def fit(self) -> np.float32:
        """ @return: in-sampler error """
        if self.w is None:
            self.w: np.ndarray = np.zeros(len(self.x[0]))

        for _ in range(self.cfg.stop):
            x, y = Regressor.select(self.x, self.y)
            self.w += self.cfg.eta * Regressor.logis(-(x @ self.w) * y) * y * x

        return self.cfg.loss(self.x, self.y, self.w)


def attribute(description: str) -> Callable:
    """ decorator for the problem function 
    to print out some descriptions """
    return lambda func: lambda *args, **kwargs: \
        (print(description), func(*args, **kwargs))


@attribute("[Problem 14]")
def problem_14() -> tuple:
    x, y = Configure.load_data("./data/train.dat")
    cfg: Configure = Configure(loss=Loss.mse)
    
    w, error = Linear(x, y, cfg).fit()

    print(f"Average in-sample error: {error:.4}")
    return w, error


@attribute("[Problem 15]")
def problem_15(error: np.float32) -> None:
    x, y = Configure.load_data("./data/train.dat")
    cfg: Configure = Configure(
        loss=Loss.mse, stop=1.01 * error
    )

    update_times: list = []
    for _ in tqdm(range(cfg.epoch)):
        times: int = Linear_SGD(x, y, cfg).fit()
        update_times.append(times)

    print(f"Average update times: {np.array(update_times).mean()}")


@attribute("[Problem 16]")
def problem_16() -> None:
    x, y = Configure.load_data("./data/train.dat")
    cfg: Configure = Configure(
        loss=Loss.ce, stop=500
    )

    errors: list[np.float32] = []
    for _ in tqdm(range(cfg.epoch)):
        error: np.float32 = Logistic_SGD(x, y, cfg).fit()
        errors.append(error)

    print(f"Average in-sample error: {np.array(errors).mean():.4}")


@attribute("[Problem 17]")
def problem_17(w: np.ndarray) -> None:
    x, y = Configure.load_data("./data/train.dat")
    cfg: Configure = Configure(
        loss=Loss.ce, stop=500, w=w
    )

    errors: list[np.float32] = []
    for _ in tqdm(range(cfg.epoch)):
        error: np.float32 = Logistic_SGD(x, y, cfg).fit()
        errors.append(error)

    print(f"Average in-sample error: {np.array(errors).mean():.4}")


@attribute("[Problem 18]")
def problem_18() -> None:
    train_x, train_y = Configure.load_data("./data/train.dat")
    valid_x, valid_y = Configure.load_data("./data/test.dat")
    cfg: Configure = Configure(loss=Loss.one)

    w, e_in = Linear(train_x, train_y, cfg).fit()
    e_out: np.float32 = Loss.one(valid_x, valid_y, w)
    
    print(f"Absolute difference: {abs(e_out - e_in):.4}")


@attribute("[Problem 19]")
def problem_19() -> None:
    train_x, train_y = Configure.load_data("./data/train.dat", 3)
    valid_x, valid_y = Configure.load_data("./data/test.dat", 3)
    cfg: Configure = Configure(loss=Loss.one)

    w, e_in = Linear(train_x, train_y, cfg).fit()
    e_out: np.float32 = Loss.one(valid_x, valid_y, w)

    print(f"Absolute difference: {abs(e_out - e_in):.4}")


@attribute("[Problem 20]")
def problem_20() -> None:
    train_x, train_y = Configure.load_data("./data/train.dat", 10)
    valid_x, valid_y = Configure.load_data("./data/test.dat", 10)
    cfg: Configure = Configure(loss=Loss.one)

    w, e_in = Linear(train_x, train_y, cfg).fit()
    e_out: np.float32 = Loss.one(valid_x, valid_y, w)

    print(f"Absolute difference: {abs(e_out - e_in):.4}")


def main() -> None:
    _, args = problem_14()
    problem_15(args[1])
    problem_16()
    problem_17(args[0])
    problem_18()
    problem_19()
    problem_20()


if __name__ == "__main__":
    main()
