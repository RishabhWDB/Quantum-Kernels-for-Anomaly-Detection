import numpy as np


def offdiagonal(K):
    return K[np.triu_indices_from(K, k=1)]


def concentration(K):
    v = offdiagonal(K)
    return {"offdiag_mean": float(v.mean()), "offdiag_var": float(v.var())}


def center(K):
    n = len(K)
    H = np.eye(n) - np.full((n, n), 1.0 / n)
    return H @ K @ H


def alignment(K, y):
    """Centred kernel-target alignment A(K, y)."""
    s = np.where(y > 0, 1.0, -1.0)
    Kc, Yc = center(K), center(np.outer(s, s))
    return float(np.sum(Kc * Yc) / (np.linalg.norm(Kc) * np.linalg.norm(Yc)))


def effective_rank(K):
    mu = np.clip(np.linalg.eigvalsh(K), 0.0, None)
    p = mu / mu.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _sqrt_psd(K):
    w, V = np.linalg.eigh(K)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def geometric_difference(K_classical, K_quantum, eps=1e-3):
    """g(K_C || K_Q) = sqrt(||sqrt(K_Q) sqrt(K_C) (K_C + eps I)^-2 sqrt(K_C) sqrt(K_Q)||_inf), traces normalised to n."""
    n = len(K_quantum)
    Kc = K_classical * n / np.trace(K_classical)
    Kq = K_quantum * n / np.trace(K_quantum)
    sq, sc = _sqrt_psd(Kq), _sqrt_psd(Kc)
    inv = np.linalg.inv(Kc + eps * np.eye(n))
    M = sq @ sc @ inv @ inv @ sc @ sq
    return float(np.sqrt(np.linalg.norm(M, 2)))


def kernel_correlation(K1, K2):
    return float(np.corrcoef(offdiagonal(K1), offdiagonal(K2))[0, 1])


def normalized_centered(K):
    Kc = center(K)
    return Kc / np.abs(Kc).max()
