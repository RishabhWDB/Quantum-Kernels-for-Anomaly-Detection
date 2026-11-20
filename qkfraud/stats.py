import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_p(a, b):
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def bootstrap_ci(diff, n_resamples=10_000, level=0.95, seed=0):
    diff = np.asarray(diff)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_resamples, len(diff)), replace=True).mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_comparison(a, b, n_resamples=10_000, seed=0):
    """Mean paired difference a - b with bootstrap CI and two-sided Wilcoxon p-value."""
    a, b = np.asarray(a), np.asarray(b)
    low, high = bootstrap_ci(a - b, n_resamples, seed=seed)
    return {"delta": float((a - b).mean()), "ci_low": low, "ci_high": high, "p": wilcoxon_p(a, b)}


def family(comparisons):
    """Holm-correct a list of comparison dicts in place."""
    for c, p in zip(comparisons, holm([c["p"] for c in comparisons])):
        c["p_holm"] = p
    return comparisons
