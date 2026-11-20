import numpy as np


def depolarizing_survival(eps, n_two_qubit):
    return (1.0 - eps) ** n_two_qubit


def depolarize(K, eps, n_two_qubit, d, train=False):
    """Global depolarizing channel on the kernel circuit: K -> p K + (1 - p) 2^-d."""
    p = depolarizing_survival(eps, n_two_qubit)
    noisy = p * K + (1.0 - p) * 2.0 ** (-d)
    if train:
        np.fill_diagonal(noisy, 1.0)
    return noisy


def sample_shots(K, shots, rng, train=False):
    """Replace estimated entries by Bin(S, K) / S; training kernels are sampled on the upper triangle and mirrored."""
    if not train:
        return rng.binomial(shots, np.clip(K, 0.0, 1.0)) / shots
    iu = np.triu_indices_from(K, k=1)
    noisy = np.eye(len(K))
    noisy[iu] = rng.binomial(shots, np.clip(K[iu], 0.0, 1.0)) / shots
    return noisy + np.triu(noisy, k=1).T


def clip_psd(K, tol=1e-8):
    w, V = np.linalg.eigh(K)
    if w.min() >= -tol:
        return K, False, float(w.min())
    return (V * np.clip(w, 0.0, None)) @ V.T, True, float(w.min())


def noisy_kernels(K_train, K_test, d, rng, shots=None, eps=0.0, n_two_qubit=0):
    if eps > 0:
        K_train = depolarize(K_train, eps, n_two_qubit, d, train=True)
        K_test = depolarize(K_test, eps, n_two_qubit, d)
    if shots:
        K_train = sample_shots(K_train, shots, rng, train=True)
        K_test = sample_shots(K_test, shots, rng)
    return K_train, K_test
