from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def stratified_folds(y, k, seed):
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


@dataclass
class Tuned:
    model: str
    params: dict
    cv_score: float
    edges: dict = field(default_factory=dict)
    table: list = field(default_factory=list)


def grid_edges(table, grid, params, best_score, tol):
    """Label every grid-edge choice as a plateau or as still improving.

    An edge value is still improving if its CV score beats the best configuration
    that uses the adjacent interior value by more than ``tol``.
    """
    edges = {}
    for name, values in grid.items():
        numeric = sorted(v for v in values if not isinstance(v, str))
        value = params[name]
        if isinstance(value, str) or len(numeric) < 2 or value not in (numeric[0], numeric[-1]):
            continue
        side, neighbour = ("min", numeric[1]) if value == numeric[0] else ("max", numeric[-2])
        alternative = max(score for p, score in table if p[name] == neighbour)
        edges[name] = {
            "value": value,
            "side": side,
            "status": "still_improving" if best_score - alternative > tol else "plateau",
        }
    return edges


def select(model, table, grid, tol):
    params, score = max(table, key=lambda row: row[1])
    return Tuned(model, params, float(score), grid_edges(table, grid, params, score, tol), table)


def _svm(C, kernel="precomputed", gamma="scale"):
    return SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced")


def svm_cv_precomputed(K, y, folds, C):
    scores = []
    for tr, va in folds:
        model = _svm(C).fit(K[np.ix_(tr, tr)], y[tr])
        scores.append(average_precision_score(y[va], model.decision_function(K[np.ix_(va, tr)])))
    return float(np.mean(scores))


def tune_C(K, y, folds, Cs, tol, model="qsvm"):
    table = [({"C": C}, svm_cv_precomputed(K, y, folds, C)) for C in Cs]
    return select(model, table, {"C": Cs}, tol)


def tune_qsvm(kernel, X, y, folds, grid, tol):
    """Joint (bandwidth, C) search; one kernel matrix per bandwidth, sliced per fold."""
    table = []
    for lam in grid["bandwidth"]:
        K = kernel(X, bandwidth=lam)
        table += [({"bandwidth": lam, "C": C}, svm_cv_precomputed(K, y, folds, C)) for C in grid["C"]]
    return select("qsvm", table, grid, tol)


def tune_rbf(X, y, folds, grid, tol):
    table = []
    for C in grid["C"]:
        for gamma in grid["gamma"]:
            scores = []
            for tr, va in folds:
                model = _svm(C, "rbf", gamma).fit(X[tr], y[tr])
                scores.append(average_precision_score(y[va], model.decision_function(X[va])))
            table.append(({"C": C, "gamma": gamma}, float(np.mean(scores))))
    return select("rbf_svm", table, grid, tol)


def _iforest(params, contamination, seed):
    return IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        contamination=contamination,
        random_state=seed,
    )


def tune_iforest(X, y, folds, grid, tol, seed):
    contamination = float(y.mean())
    table = []
    for n in grid["n_estimators"]:
        for m in grid["max_samples"]:
            params = {"n_estimators": n, "max_samples": m}
            scores = []
            for tr, va in folds:
                model = _iforest(params, contamination, seed).fit(X[tr])
                scores.append(average_precision_score(y[va], -model.score_samples(X[va])))
            table.append((params, float(np.mean(scores))))
    return select("isolation_forest", table, grid, tol)


def qsvm_predict(K_train, y, C, K_tests):
    model = _svm(C).fit(K_train, y)
    return {k: (model.decision_function(K), model.predict(K)) for k, K in K_tests.items()}


def rbf_predict(X, y, params, tests):
    model = _svm(params["C"], "rbf", params["gamma"]).fit(X, y)
    return {k: (model.decision_function(T), model.predict(T)) for k, T in tests.items()}


def iforest_predict(X, y, params, seed, tests):
    model = _iforest(params, float(y.mean()), seed).fit(X)
    return {k: (-model.score_samples(T), (model.predict(T) == -1).astype(int)) for k, T in tests.items()}
