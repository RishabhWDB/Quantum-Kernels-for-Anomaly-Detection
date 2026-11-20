import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RECALL_GRID = np.linspace(0.0, 1.0, 101)


def recall_at_fpr(y, scores, max_fpr):
    fpr, tpr, _ = roc_curve(y, scores)
    return float(tpr[fpr <= max_fpr].max())


def precision_at_k(y, scores, k=None):
    k = int(y.sum()) if k is None else k
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y[top].mean())


def best_f1(y, scores):
    precision, recall, _ = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())


def operating_point(y, scores, recall):
    """Precision and false positives per 1,000 transactions at the highest threshold reaching ``recall``."""
    precision, rec, thresholds = precision_recall_curve(y, scores)
    i = np.flatnonzero(rec[:-1] >= recall).max()
    false_positives = int(((scores >= thresholds[i]) & (y == 0)).sum())
    return float(precision[i]), 1000.0 * false_positives / len(y)


def interpolated_pr(y, scores, grid=RECALL_GRID):
    precision, recall, _ = precision_recall_curve(y, scores)
    return [float(precision[recall >= r].max()) for r in grid]


def evaluate(y, scores, predictions, recalls=(), curve=False):
    out = {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_fpr_1": recall_at_fpr(y, scores, 0.01),
        "recall_at_fpr_5": recall_at_fpr(y, scores, 0.05),
        "precision_at_k": precision_at_k(y, scores),
        "best_f1": best_f1(y, scores),
        "mcc": float(matthews_corrcoef(y, predictions)),
    }
    for r in recalls:
        p, fp = operating_point(y, scores, r)
        out[f"precision_at_recall_{round(100 * r)}"] = p
        out[f"fp_per_1k_at_recall_{round(100 * r)}"] = fp
    if curve:
        out["pr_curve"] = interpolated_pr(y, scores)
        out["positive_rate"] = float(np.mean(y))
    return out
