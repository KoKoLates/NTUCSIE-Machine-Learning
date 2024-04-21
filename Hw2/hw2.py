# -*- coding: utf-8 -*-

import random
import numpy as np

from tqdm import tqdm
from typing import Callable


class Data(object):

    @staticmethod
    def generator(num: int) -> tuple[np.ndarray, np.ndarray]:
        """data generator"""
        x: np.ndarray = np.random.uniform(-1, 1, num)
        y: np.ndarray = np.where(x > 0, 1, -1)

        return x, y

    @staticmethod
    def noise(y: np.ndarray, tua: float | int) -> None:
        """ adding the noise with a fixed distribution 
        to both train data and test (valid) data 
        @param y: output of target function of data
        @param tua: noising distribution for the data 
        """
        if not tua: return

        for idx, value in enumerate(y):
            if random.randint(1, tua * 100) == 1:
                y[idx] = -value


class DecisionStump(object):
    def __init__(
        self,
        train: list[tuple[np.ndarray, np.ndarray]],
        valid: list[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """ processing the decision stump algorithms on both
        in-sample and out-of-sample data at the same time """
        self.train: list = train
        self.valid: list = valid

        ## processing
        self.e_i: float = self._fit() # in-sample error
        self.e_o: float = self._out_of_sample_error() # out-of-sample error
    
    def _fit(self) -> float:
        """ training decision stump with in-sample data 
        and calculate the in-sample error """

        ## calculate s and theta for hypothesis
        self.train = sorted(self.train)

        theta_list: list = [-1]
        previous_x: float = 0
        for idx, (x, _) in enumerate(self.train):
            if not idx: 
                previous_x = x
                continue

            if previous_x == x: 
                continue

            theta_list.append((x + previous_x) / 2)
            previous_x = x     

        error: float = float("inf")
        for theta in theta_list:
            for s in (-1, 1):
                e: float = sum(1 for x, y in self.train if s * (x - theta) * y < 0)
                
                if (e := e / len(self.train)) >= error: continue

                error = e
                self.s = s
                self.theta = theta

        return error
    
    def _out_of_sample_error(self) -> float:
        """ based on the hypothesis obtain from in-sample data
        calculate the out-of-sample error """

        error: float = 0
        for x, y in self.valid:
            error += 0 if self.s * (x - self.theta) * y > 0 else 1

        error /= len(self.valid)
        return error
    
    def result(self) -> tuple:
        """ calculating the difference and return the results of decision 
        stump with in-sample error and out-of-sample error.
        @return: triplet of `[e_in, e_out, difference]`
        """
        diff: float = self.e_o - self.e_i
        return (self.e_i, self.e_o, diff)


def attribute(fun: Callable) -> Callable:
    """function decorator to display some informations
    @param fun: callable function for the decorator.
    @return: the wrapped function.
    """

    def decorate(params: tuple[float, int]) -> None:
        """show the problem parameters
        @param params: a tuple store `tua` and train size.
        """
        tua: float = params[0] if isinstance(params[0], float) else 0.0

        print(f"Problem {decorate.nums}\n{'-'*20}")
        print(f"tua: {tua} | size: {params[1]:03}")
        decorate.nums += 1
        fun(params)

    decorate.nums = 16
    return decorate


@attribute
def problem(params: tuple[float, int]) -> None:
    errors: list = []
    tua, train_size = params
    for _ in tqdm(range(10000)):
        x_train, y_train = Data.generator(train_size)
        x_valid, y_valid = Data.generator(10000)

        # add noise for the labels
        Data.noise(y_train, tua)
        Data.noise(y_valid, tua)

        decision_stump = DecisionStump(
            list(zip(x_train, y_train)), 
            list(zip(x_valid, y_valid))    
        )

        _, _, error = decision_stump.result()
        errors.append(error)

    print(np.array(errors).mean())


def main() -> None:
    questions: dict[int, tuple] = {
        16: (0, 2),
        17: (0, 20),
        18: (0.1, 2),
        19: (0.1, 20),
        20: (0.1, 200),
    }

    for _, param in questions.items():
        problem(param)


if __name__ == "__main__":
    main()
