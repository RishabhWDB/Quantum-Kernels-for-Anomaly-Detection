from dataclasses import dataclass

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    X = df.drop(columns="Class").to_numpy(dtype=float)
    y = df["Class"].to_numpy(dtype=int)
    return X, y


@dataclass
class Split:
    """Disjoint stratified index pools for one seed.

    The training pool is ordered so that the first ``n`` rows of each class form
    nested training sets; the base training set (``n_train``) comes first and the
    extra rows used for larger learning-curve sizes follow it.
    """

    train_fraud: np.ndarray
    train_normal: np.ndarray
    test_fraud: np.ndarray
    test_normal: np.ndarray
    prevalence_normal: np.ndarray
    fraud_fraction: float

    def train(self, n):
        k = round(n * self.fraud_fraction)
        if k > len(self.train_fraud) or n - k > len(self.train_normal):
            raise ValueError(f"training size {n} exceeds the reserved pool")
        return np.concatenate([self.train_fraud[:k], self.train_normal[: n - k]])

    def test(self):
        return np.concatenate([self.test_fraud, self.test_normal])

    def prevalence_test(self):
        return np.concatenate([self.test_fraud, self.test_normal, self.prevalence_normal])


def make_split(y, seed, n_train, n_test, fraud_fraction, n_train_max, prevalence_normals):
    rng = np.random.default_rng(seed)
    fraud = rng.permutation(np.flatnonzero(y == 1))
    normal = rng.permutation(np.flatnonzero(y == 0))

    f_test = round(n_test * fraud_fraction)
    n_test_normal = n_test - f_test
    f_train = round(n_train_max * fraud_fraction)
    n_train_normal = n_train_max - f_train
    n_extra_normal = prevalence_normals - n_test_normal
    if f_test + f_train > len(fraud) or n_test_normal + n_train_normal + n_extra_normal > len(normal):
        raise ValueError("not enough transactions for the requested split")

    test_fraud, fraud = fraud[:f_test], fraud[f_test:]
    test_normal, normal = normal[:n_test_normal], normal[n_test_normal:]
    train_fraud = fraud[:f_train]
    train_normal, normal = normal[:n_train_normal], normal[n_train_normal:]
    prevalence_normal = normal[:n_extra_normal]
    return Split(train_fraud, train_normal, test_fraud, test_normal, prevalence_normal, fraud_fraction)
