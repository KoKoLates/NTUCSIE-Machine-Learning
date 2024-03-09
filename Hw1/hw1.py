# -*- coding: utf-8 -*-

import random
import argparse
import numpy as np

from tqdm import tqdm


class Perceptron(object):
    def __init__(self, x: np.ndarray, y: np.ndarray, bias: int = 1) -> None:
        """ perceptron learning algorithm 
        @param x: a series of traing data
        @param y: a series of labels
        @param bias: a bias value added to the data
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.add_bias(bias)

        self.w: np.ndarray = np.zeros(self.x.shape[1])

    def fit(self) -> tuple:
        correct, update = 0, 0
        while correct < self.x.shape[0] * 5:
            n: int = self.get_random()
            x_n, y_n = self.x[n], self.y[n]

            predict: int = Perceptron.sign(np.dot(self.w.T, x_n))
            if predict == y_n: # correct prediction
                correct += 1
                continue

            self.w += y_n * x_n
            update += 1
            correct = 0

        return update, self.w[0]

    def add_bias(self, bias: int) -> None:
        """ add bias in front of data """
        self.x = np.insert(self.x, 0, bias, axis=1)

    def get_random(self) -> int:
        """ pick random one in a series of data """
        return random.randint(0, self.x.shape[0] - 1)

    @staticmethod
    def sign(value: float) -> int:
        return 1 if value > 0 else -1

    @staticmethod
    def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
        with open(file_path, "r") as file:
            data: list[list[float]] = []
            for line in file.readlines():
                data.append([float(value) for value in line.split()])

        x: np.ndarray = np.array([v[:-1] for v in data])
        y: np.ndarray = np.array([v[-1]  for v in data])
        return x, y


def problem_16() -> None:
    x, y = Perceptron.load_data(args.f)

    updates: list = []
    for _ in tqdm(range(1000)):
        p: Perceptron = Perceptron(x, y, 1)
        update, _ = p.fit()

        updates.append(update)

    print(f"-*- median of update times: {sorted(updates)[500]}")


def problem_17() -> None:
    x, y = Perceptron.load_data(args.f)

    weights: list = []
    for _ in tqdm(range(1000)):
        p: Perceptron = Perceptron(x, y, 1)
        _, w0 = p.fit()

        weights.append(w0)

    print(f"-*- median of bias weights: {sorted(weights)[500]}")


def problem_18() -> None:
    x, y = Perceptron.load_data(args.f)

    updates: list = []
    for _ in tqdm(range(1000)):
        p: Perceptron = Perceptron(x, y, 10)
        update, _ = p.fit()

        updates.append(update)

    print(f"-*- median of update times: {sorted(updates)[500]}")


def problem_19() -> None:
    x, y = Perceptron.load_data(args.f)

    updates: list = []
    for _ in tqdm(range(1000)):
        p: Perceptron = Perceptron(x, y, 0)
        update, _ = p.fit()

        updates.append(update)

    print(f"-*- median of update times: {sorted(updates)[500]}")


def problem_20() -> None:
    x, y = Perceptron.load_data(args.f)
    x /= 4 ## scale down by 4

    updates: list = []
    for _ in tqdm(range(1000)):
        p: Perceptron = Perceptron(x, y, 0)
        update, _ = p.fit()

        updates.append(update)

    print(f"-*- median of update times: {sorted(updates)[500]}")


def main() -> None:
    questions: dict = {
        16: problem_16,
        17: problem_17,
        18: problem_18,
        19: problem_19,
        20: problem_20,
    }

    try:
        print(f"Problem {args.q}\n{'-'*20}")
        questions[args.q]()
    except KeyError as error:
        print(f"[error] questions list has no key: {error}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, required=True, 
                        help="the file path of data")
    parser.add_argument("-q", type=int, required=True, 
                        help="the question number from 16 to 20")
    args: argparse.Namespace = parser.parse_args()
    main()
