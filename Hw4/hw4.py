# -*- coding: utf-8 -*-

import math

import argparse
import numpy as np

from typing import Callable
from dataclasses import dataclass

from liblinear.python.liblinear.liblinearutil import *


@dataclass
class Data(object):

    @staticmethod
    def load_data(
        file_path: str, Q: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """load in the dataset for indicate file path
        @param file_path: file path of dataset files
        @param Q: nonlinear transformation Q-th order
        """
        with open(file_path, "r") as file:
            data: list[list[float]] = []
            for line in file.readlines():
                data.append([float(value) for value in line.split()])

        x: np.ndarray = np.array([v[:-1] for v in data])
        y: np.ndarray = np.array([1 if v[-1] > 0 else 0 for v in data])

        if Q is not None:
            x = Data.transform(x, Q)

        ## adding the bias to the weights
        x = np.insert(x, 0, 1, axis=1)

        return x, y

    @staticmethod
    def transform(x: np.ndarray, Q: int = 2) -> np.ndarray:
        """nonlinear transformation"""
        new_x: list = []
        for sample in x:
            t: list = []
            for i in range(len(sample)):
                for j in range(i, len(sample)):
                    t.append(sample[i] * sample[j])
            new_x.append(t)

        return np.hstack((x, np.array(new_x)))

    @staticmethod
    def cross_validate(x: np.ndarray, y: np.ndarray, index: int) -> tuple:
        """split the dataset in folder by cross validation
        @param x: features data
        @param y: labeled data
        @param index: folder's index
        @return: `(train_x, train_y, valid_x, valid_y)`
        """
        return (
            np.vstack((x[: index * 40], x[(index + 1) * 40 :])),
            np.hstack((y[: index * 40], y[(index + 1) * 40 :])),
            x[index * 40 : (index + 1) * 40],
            y[index * 40 : (index + 1) * 40],
        )


class Loss(object):
    """loss function static class"""

    @staticmethod
    def one(y: np.ndarray, y_hat: np.ndarray) -> np.float32:
        """1/0 loss function
        @param y: the real target output
        @param y_hat: the predict output
        @return: average 1/0 error (true/false count)
        """
        error: np.ndarray = abs(y_hat - y)
        return error.mean()


def attribute(text: str) -> Callable:
    """decorator to print out the description
    @param text: descriptions"""
    return lambda func: lambda: print(f"{text}\n{'-'*20}") or func()


@attribute("[Problem 16]")
def problem_16() -> None:
    data_x, data_y = Data.load_data("./data/train.dat", 2)
    test_x, test_y = Data.load_data("./data/test.dat", 2)

    errors: dict = {}
    for log_lambda in (-4, -2, 0, 2, 4):
        model = train(
            problem(data_y, data_x),
            f"-s 0 -c {1 / 10 ** log_lambda / 2} -e 0.000_001 -q",
        )

        p, _, _ = predict(test_y, test_x, model, "-b 1")
        error: np.float32 = Loss.one(test_y, p)

        errors.update({error: 10**log_lambda})
        print(f"Eout: {f'{error:.4}'.ljust(7, ' ')} lambda: {10 ** log_lambda}\n")

    best: float = errors[min(errors.keys())]
    print(f"Best Lambda is {best}\nLog(Best Lambda): {math.log10(best)}")


@attribute("[Problem 17]")
def problem_17() -> None:
    data_x, data_y = Data.load_data("./data/train.dat", 2)

    errors: dict = {}
    for log_lambda in (-4, -2, 0, 2, 4):
        model = train(
            problem(data_y, data_x),
            f"-s 0 -c {1 / 10 ** log_lambda / 2} -e 0.000_001 -q",
        )

        p, _, _ = predict(data_y, data_x, model, "-b 1")
        error: np.float32 = Loss.one(data_y, p)

        errors.update({error: 10**log_lambda})
        print(f"Ein: {f'{error:.4}'.ljust(7, ' ')} lambda: {10 ** log_lambda}\n")

    best: float = errors[min(errors.keys())]
    print(f"Best Lambda is {best}\nLog(Best Lambda): {math.log10(best)}")


@attribute("[Problem 18]")
def problem_18() -> None:
    train_x, train_y = Data.load_data("./data/train.dat", 2)
    valid_x, valid_y = train_x[120:], train_y[120:]
    train_x, train_y = train_x[:120], train_y[:120]

    errors: dict = {}
    for log_lambda in (-4, -2, 0, 2, 4):
        model = train(
            problem(train_y, train_x),
            f"-s 0 -c {1 / 10 ** log_lambda / 2} -e 0.000_001 -q",
        )

        p, _, _ = predict(valid_y, valid_x, model, "-b 1")
        error: np.float32 = Loss.one(valid_y, p)

        errors.update({error: (10**log_lambda, model)})
        print(f"Eval: {f'{error:.4}'.ljust(7, ' ')} lambda: {10 ** log_lambda}\n")

    ## testing error measurement
    test_x, test_y = Data.load_data("./data/test.dat", 2)
    p, _, _ = predict(test_y, test_x, errors[min(errors.keys())][1], "-b 1 -q")
    error: float = Loss.one(test_y, p)

    print(f"Best Lambda is {errors[min(errors.keys())][0]}\nEout(best): {error:.4}")


@attribute("[Problem 19]")
def problem_19() -> None:
    data_x, data_y = Data.load_data("./data/train.dat", 2)
    test_x, test_y = Data.load_data("./data/test.dat", 2)

    model = train(problem(data_y, data_x), "-s 0 -c 50 -e 0.000_001 -q")

    p, _, _ = predict(test_y, test_x, model, "-b 1 -q")
    error: np.float32 = Loss.one(test_y, p)

    print(f"Best Lambda is 0.01\nEout(best): {error:.4}")


@attribute("[Problem 20]")
def problem_20() -> None:
    data_x, data_y = Data.load_data("./data/train.dat", 2)

    errors: dict = {}
    for log_lambda in (-4, -2, 0, 2, 4):
        error: float = 0
        for index in range(5):
            train_x, train_y, valid_x, valid_y = Data.cross_validate(
                data_x, data_y, index
            )

            model = train(
                problem(train_y, train_x),
                f"-s 0 -c {1 / 10 ** log_lambda / 2} -e 0.000_001 -q",
            )

            p, _, _ = predict(valid_y, valid_x, model, "-b 1 -q")
            error += Loss.one(valid_y, p)

        errors.update({error / 5: (10**log_lambda)})
        print(f"Eval: {f'{(error / 5):.4}'.ljust(7, ' ')} lambda: {10 ** log_lambda}\n")

    best: float = min(errors.keys())
    print(f"Best Lambda is {errors[best]}\nEval(best): {best:.4}")


def main() -> None:
    questions: dict = {  ## the problem list
        16: problem_16,
        17: problem_17,
        18: problem_18,
        19: problem_19,
        20: problem_20,
    }

    index: int = args.q
    if index is not None:
        try:
            questions[index]()
            return
        except KeyError as error:
            print(f"[error] question list has no problem {error}")

    for problem in questions.values():
        problem()
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        type=int,
        choices=[16, 17, 18, 19, 20],
        help="select specific quesion for execution.",
    )
    args = parser.parse_args()
    main()
