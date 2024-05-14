# -*- coding: utf-8 -*-

import argparse
import numpy as np

from tqdm import tqdm
from typing import Callable
from libsvm.python.libsvm.svmutil import *


class label_data(object):
    """special class just for data labeling"""

    def __init__(self, y: np.ndarray, class_: int) -> None:
        """constructor for data labeling
        @param y: input data to be labeled with class
        @param class_: the class to be classify
        """
        self.y: np.ndarray = y
        self.class_: int = class_

    def __call__(self) -> np.ndarray:
        """__call__ method implement for label indicated class"""
        self.output: list[int] = []

        for value in self.y:
            self.output.append(1 if value == self.class_ else 0)

        return np.array((self.output))


def attribute(text: str) -> Callable:
    """decorator to print out the description
    @param text: descriptions"""
    return lambda func: lambda: print(f"{text}\n{'-'*20}") or func()


@attribute("[Problem 15]")
def problem_15() -> None:
    y, x = svm_read_problem("./data/train.dat")
    y: np.ndarray = label_data(y, 3)()

    model: svm_model = svm_train(
        svm_problem(y, x), "-t 0 -c 10 -q"
    )

    alpha = list(map(lambda x: x[0], model.get_sv_coef()))
    SV: list[dict] = model.get_SV()

    sv_val: list[list] = []
    for s in SV:
        sv: list = list(s.keys())
        miss: list = [i for i in range(1, 37) if i not in sv]

        target: list = list(s.values())
        for i in miss:
            target.insert(i - 1, 0)
        sv_val.append(target)

    w: np.ndarray = np.array(alpha).dot(sv_val)
    norm_w: float = np.sqrt(np.power(w, 2).sum())
    print(f"*** the norm of weights: {norm_w:.4} ***")


@attribute("[Problem 16]")
def problem_16() -> None:
    error: list = []
    for i in range(1, 6):
        print(f"- {i} versus not {i} -")

        y, x = svm_read_problem("./data/train.dat")
        y: np.ndarray = label_data(y, i)()

        model: svm_model = svm_train(
            svm_problem(y, x), "-t 1 -c 10 -g 1 -d 2 -r 1 -q"
        )

        acc, _, _ = svm_predict(y, x, model)[1]
        error.append(1 - 0.01 * acc)

    print(f"** minimum Ein: {min(error):.4} **")


@attribute("[Problem 17]")
def problem_17() -> None:
    SV: list[int] = []
    for i in range(1, 6):
        y, x = svm_read_problem("./data/train.dat")
        y: np.ndarray = label_data(y, i)()

        model: svm_model = svm_train(
            svm_problem(y, x), "-t 1 -c 10 -g 1 -d 2 -r 1 -q"
        )
        SV.append(model.get_nr_sv())

    print(f"** maximum n(support vector): {max(SV)} **")


@attribute("[Problem 18]")
def problem_18() -> None:
    y_data, x_data = svm_read_problem("./data/train.dat")
    y_test, x_test = svm_read_problem("./data/test.dat")

    y_data: np.ndarray = label_data(y_data, 6)()
    y_test: np.ndarray = label_data(y_test, 6)()

    error: dict[float, float] = {}
    for C in (0.01, 0.1, 1, 10, 100):
        print(f"- C: {C} -")
        model: svm_model = svm_train(
            svm_problem(y_data, x_data), f"-t 2 -c {C} -g 10 -q"
        )

        acc, _, _ = svm_predict(y_test, x_test, model)[1]
        error.update({1 - 0.01 * acc: C})
        print(f"Eout = {1 - 0.01 * acc:.4}")

    min_error: float = min(error.keys())
    c: float = error[min_error]
    print(f"** minimum Eout: {min_error:.4} when C is: {c} **")


@attribute("[Problem 19]")
def problem_19() -> None:
    y_data, x_data = svm_read_problem("./data/train.dat")
    y_test, x_test = svm_read_problem("./data/test.dat")

    y_data: np.ndarray = label_data(y_data, 6)()
    y_test: np.ndarray = label_data(y_test, 6)()

    error: dict[float, float] = {}
    for G in (0.1, 1, 10, 100, 1000):
        print(f"- G: {G} -")
        model: svm_model = svm_train(
            svm_problem(y_data, x_data), f"-t 2 -c 0.1 -g {G} -q"
        )

        acc, _, _ = svm_predict(y_test, x_test, model)[1]
        error.update({1 - 0.01 * acc: G})
        print(f"Eout = {1 - 0.01 * acc:.4}")

    min_error: float = min(error.keys())
    c: float = error[min_error]
    print(f"** minimum Eout: {min_error:.4} when G is: {c} **")


@attribute("[Problem 20]")
def problem_20() -> None:
    y, x = svm_read_problem("./data/train.dat")
    y: np.ndarray = label_data(y, 6)()
    
    count: dict[float, int] = {
        0.1: 0, 1: 0, 10: 0, 100: 0, 1000: 0 
    }

    def get_valid(x: np.ndarray, y: np.ndarray, N: int = 200) -> tuple:
        """shuffle the input data and split into train and valid dataset
        @param x: data feature
        @param y: data class after labeled
        @param N: size of validation set
        """
        np.random.shuffle(x)
        np.random.shuffle(y)
        return x[N:], y[N:], x[:N], y[:N]

    for _ in tqdm(range(1000)):
        x_train, y_train, x_valid, y_valid = get_valid(x, y, 200)

        error: dict[float, float] = {}
        for G in (0.1, 1, 10, 100, 1000):
            model: svm_model = svm_train(
                svm_problem(y_train, x_train), f"-t 2 -c 0.1 -g {G} -q"
            )

            acc, _, _ = svm_predict(y_valid, x_valid, model, "-q")[1]
            error.update({1 - 0.01 * acc: G})

        index: float = error[min(error.keys())]
        count[index] += 1

    print(f"** number of gamma selection: {count} **")


def main() -> None:
    questions: dict[int, Callable] = {
        15: problem_15, 16: problem_16, 17: problem_17,
        18: problem_18, 19: problem_19, 20: problem_20
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
        choices=[15, 16, 17, 18, 19, 20],
        help="select specific question for execution.",
    )
    args = parser.parse_args()
    main()
