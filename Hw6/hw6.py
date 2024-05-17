# -*- coding: utf-8 -*-

import argparse
import numpy as np

from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class Data(object):
    """dataclass of dataset"""

    @staticmethod
    def load_data(file_path: str) -> tuple:
        """load the data from indicated files"""
        with open(file_path, "r") as file:
            data: list[list[float]] = []
            for line in file.readlines():
                data.append([float(value) for value in line.split()])

        x: np.ndarray = np.array([v[:-1] for v in data])
        y: np.ndarray = np.array([v[-1:] for v in data])
        return x, y


@dataclass
class Loss(object):
    """loss function data class"""

    @staticmethod
    def one(y: np.ndarray, y_hat: np.ndarray) -> float:
        """one/zero (0/1) error measurement
        @param y: true value with label
        @param y_hat: model prediction
        """
        error: np.ndarray = y != y_hat
        return error.sum() / len(y)


@dataclass
class Node(object):
    """node data that store the basic attirbute of node
    and some utils function for internal node operation"""

    label: Optional[int] = None
    index: Optional[int] = None
    theta: Optional[float] = None

    l: Optional["Node"] = None
    r: Optional["Node"] = None

    @staticmethod
    def branching(x: np.ndarray, y: np.ndarray) -> tuple:
        """finding the best criteria and branching the data
        @param x: input data
        @param y: label data"""
        n_features: int = x.shape[1]
        best_impurity: float = float("inf")

        def get_theta(x: list) -> np.ndarray:
            list_: list[float] = []
            for i in range(1, len(x)):
                if x[i] == x[i - 1]:
                    continue

                list_.append((x[i] - x[i - 1]) / 2)

            return np.array(list_)

        for i in range(n_features):
            x_, y_ = zip(*sorted(zip(x[:, i], y)))
            theta_list: np.ndarray = get_theta(x_)

            for theta in theta_list:
                s: float = 0
                y_list: list = Node._decision_stump(np.array(x_), np.array(y_), theta)

                for c in range(2):
                    impurity: float = Node._Gini(y_list[c])
                    s += len(y_list[c]) * impurity

                if s < best_impurity:
                    best_impurity = s
                    best_i: int = i
                    best_theta: float = theta

        return best_i, best_theta

    @staticmethod
    def _decision_stump(x: np.ndarray, y: np.ndarray, theta: float) -> list[np.ndarray]:
        """decision stump hypothesis
        @param theta: partition threshold """
        mask1: np.ndarray = x < theta
        mask2: np.ndarray = x > theta
        return [y[mask1], y[mask2]]

    @staticmethod
    def _Gini(y: np.ndarray) -> float:
        """calculate the gini idnex for the impurity"""
        if len(y) == 0:
            return 1

        s: float = 0
        for i in (1, -1):
            correct: np.ndarray = y == i
            s += (correct.sum() / (len(y) + 10e-8)) ** 2

        return 1 - s


class CART(object):
    """Classification and Regression Tree (CART)"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """public method of classification and regression tree for
        bulding the tree recursively with input features and labeled data
        @param x: input data with features
        @param y: label of input dataset
        """
        # building tree recursively with root node
        self.root: Node = self._build_tree(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """public method of classification and regression tree for
        predict the output label of input data
        @param x: input testing dataset
        @return: `numpy.ndarray()` of label of each testing data
        """
        return np.array([self._traverse(i, self.root) for i in x])

    def _build_tree(self, x: np.ndarray, y: np.ndarray) -> Node:
        """recursive method of building tree
        @param x: splitted input data
        @param y: splitted label data
        @return: leaf node that contain the label for predicting only
        """
        if self._is_leaf(x, y):
            return Node(label=y[0])

        index, theta = Node.branching(x, y)
        x1, y1, x2, y2 = self._split_data(x, y, index, theta)

        return Node(
            index=index,
            theta=theta,
            l=self._build_tree(x1, y1),
            r=self._build_tree(x2, y2),
        )

    def _is_leaf(self, x: np.ndarray, y: np.ndarray) -> bool:
        """boolean verifier for the termination criteria of full-grown tree
        @param x: splitted input data
        @param y: splitted label data
        @return: the current data is terminated for the tree or not
        (is the leaf or not)
        """
        return np.all(y == y[0]) or np.all(x == x[0,:])

    def _split_data(
        self, x: np.ndarray, y: np.ndarray, i: int, theta: np.float32
    ) -> tuple:
        """split the input data and labeled data into two partition based on
        the index and threshold criteria for the left and right sub-tree
        @param x: non-splitted input data
        @param y: non-splitted label data
        @param i: features of x for the partition
        @param theta: threshold of partition
        @return: two dataset of left and right tree
        """
        mask1: np.ndarray = x[:, i] < theta
        mask2: np.ndarray = x[:, i] > theta
        return x[mask1], y[mask1], x[mask2], y[mask2]

    def _traverse(self, x: np.ndarray, node: Node) -> int:
        """according to the input, traverse the full-grown tree
        with corresponding feature and threshold on each node
        @param x: testing data
        @param ndoe: the current internal node (decision stump)
        @return: predict label
        """
        if node.label is not None:
            return node.label

        index, theta = node.index, node.theta
        return self._traverse(x, node.l if x[index] < theta else node.r)


def attribute(text: str) -> Callable:
    """decorator to print out the description
    @param text: descriptions"""
    return lambda func: lambda: print(f"{text}\n{'-'*20}") or func()


def bagging(x: np.ndarray, y: np.ndarray, n: int) -> tuple:
    """bootstrap algorithm that randomly sampling
    @param x: original dataset 
    @param y: original label data
    @param n: bootstrap dataset size
    @return: bootstrap `(x, y)` and data not sampled in bootstrap
    """
    x_, y_ = [], []
    not_use: np.ndarray = np.full(len(y), 1)
    for _ in range(n):
        i: int =np.random.randint(0, len(y) - 1)
        x_.append(x[i])
        y_.append(y[i])
        not_use[i] = 0
    
    return np.array(x_), np.array(y_), not_use


@attribute("[Problem14]")
def problem_14() -> None:
    x_data, y_data = Data.load_data("./data/train.dat")
    x_test, y_test = Data.load_data("./data/test.dat")

    tree: CART = CART().fit(x_data, y_data)
    y_hat: np.ndarray = tree.predict(x_test)

    print(f"Eout: {Loss.one(y_test, y_hat):.4f}")


@attribute("[Problem15]")
def problem_15() -> None:
    x_data, y_data = Data.load_data("./data/train.dat")
    x_test, y_test = Data.load_data("./data/test.dat")

    errors: float = 0
    for _ in range(2000):
        x_bag, y_bag, _ = bagging(x_data, y_data, len(y_data) // 2)

        tree: CART = CART().fit(x_bag, y_bag)
        errors += Loss.one(
            y_test, tree.predict(x_test)
        )

    print(f"Average Eout: {errors / 2000:.4f}")


@attribute("[Problem16]")
def problem_16() -> None:
    x_data, y_data = Data.load_data("./data/train.dat")

    Gin: np.ndarray = np.zeros(len(y_data))
    for _ in range(2000):
        x_bag, y_bag, _ = bagging(x_data, y_data, len(y_data) // 2)

        tree: CART = CART().fit(x_bag, y_bag)
        Gin += tree.predict(x_data)

    y_hat: list = []
    for i in Gin:
        y_hat.append(1 if i > 0 else -1)

    print(f"Ein(G): {Loss.one(y_data, y_hat):.4f}")


@attribute("[Problem17]")
def problem_17() -> None:
    x_data, y_data = Data.load_data("./data/train.dat")
    x_test, y_test = Data.load_data("./data/test.dat")

    Gout: np.ndarray = np.zeros(len(y_data))
    for _ in range(2000):
        x_bag, y_bag, _ = bagging(x_data, y_data, len(y_data) // 2)

        tree: CART = CART().fit(x_bag, y_bag)
        Gout += tree.predict(x_test)

    y_hat: list = []
    for i in Gout:
        y_hat.append(1 if i > 0 else -1)

    print(f"Eout(G): {Loss.one(y_test, y_hat):.4f}")


@attribute("[Problem18]")
def problem_18() -> None:
    x_data, y_data = Data.load_data("./data/train.dat")

    G_minus: np.ndarray = np.full(len(y_data), np.inf)
    for _ in range(2000):
        x_bag, y_bag, not_use = bagging(x_data, y_data, len(y_data) // 2)

        tree: CART = CART().fit(x_bag, y_bag)
        y_hat: np.ndarray = tree.predict(x_data)

        for i, y in zip(np.arange(len(y))[not_use], y_hat[not_use]):
            if G_minus[i] == np.inf:
                G_minus[i] = y
                continue

            G_minus[i] += y

    G_minus_list: list = [
        1 if G > 0 else -1 if G != np.inf else -1 for G in G_minus
    ]

    print(f"Ein(G-): {Loss.one(y_data, G_minus_list):.4f}")


def main() -> None:
    questions: dict[int, Callable] = {
        14: problem_14, 
        15: problem_15, 
        16: problem_16, 
        17: problem_17,
        18: problem_18
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
        choices=[14, 15, 16, 17, 18],
        help="select specific question for execution.",
    )
    args = parser.parse_args()
    main()
