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
        y: np.ndarray = np.sign(x)

        return x, y

    @staticmethod
    def noise(y: np.ndarray, tua: float | int) -> None:
        if not tua:
            return

        for idx, value in enumerate(y):
            if random.randint(1, tua * 100) == 1:
                y[idx] = -value


class DecisionStump(object):
    def __init__(
        self,
        train: tuple[np.ndarray, np.ndarray],
        valid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        self.train: tuple[np.ndarray, np.ndarray] = train
        self.valid: tuple[np.ndarray, np.ndarray] = valid

        ## process

        ## calculate `e_in` and `e_out`
        self.e_in: float = 1
        self.e_out: float = 3

    
    def result(self) -> tuple:
        """ calculating the error and return the results
        @return: triplet of `[E_in, E_out, error]`
        """
        error: float = self.e_out - self.e_in
        return self.e_in, self.e_out, error


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
            (x_train, y_train), (x_valid, y_valid)    
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
