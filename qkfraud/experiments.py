import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from . import data, diagnostics, features, metrics, models, noise
from .quantum import KernelCache, QuantumKernel, circuit_resources


@dataclass
class Prepared:
    seed: int
    X: np.ndarray
    y: np.ndarray
    tests: dict
    y_tests: dict
    folds: list
    variance: float = 1.0


class Context:
    def __init__(self, cfg, seeds=None):
        self.cfg = cfg
        self.seeds = list(seeds) if seeds else list(cfg["seeds"])
        self.X, self.y = data.load(cfg["data_path"])
        self.cache = KernelCache(cfg["cache_dir"], cfg["cache_max_entries"])
        self.results_dir = Path(cfg["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tol = cfg["edge_tolerance"]
        self.grids = cfg["grids"]

    def exp(self, name):
        return self.cfg["experiments"][name]

    def kernel(self, d, kind=None, reps=None):
        fm = self.cfg["feature_map"]
        return QuantumKernel(kind or fm["kind"], d, reps or fm["reps"], fm["entanglement"], self.cache)

    def split(self, seed):
        s = self.cfg["sampling"]
        n_max = max([s["n_train"], *self.exp("learning_curve")["train_sizes"]])
        return data.make_split(
            self.y, seed, s["n_train"], s["n_test"], s["fraud_fraction"], n_max, s["prevalence_normals"]
        )

    def prepare(self, seed, d=None, n_train=None, prevalence=False):
        split = self.split(seed)
        train = split.train(n_train or self.cfg["sampling"]["n_train"])
        test_idx = {"test": split.test()}
        if prevalence:
            test_idx["prevalence"] = split.prevalence_test()
        X_train = self.X[train]
        tests = {k: self.X[v] for k, v in test_idx.items()}
        if d is None:
            X_train, tests = features.standardized(X_train, tests)
            variance = 1.0
        else:
            X_train, tests, variance = features.pca_angles(X_train, tests, d)
        y_train = self.y[train]
        return Prepared(
            seed,
            X_train,
            y_train,
            tests,
            {k: self.y[v] for k, v in test_idx.items()},
            models.stratified_folds(y_train, self.cfg["cv_folds"], seed),
            variance,
        )

    def save(self, name, records, **extra):
        payload = {"experiment": name, "seeds": self.seeds, "records": records, **extra}
        with open(self.results_dir / f"{name}.json", "w") as f:
            json.dump(payload, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def tune_and_predict(ctx, p, model, d=None, kind=None, reps=None):
    if model == "qsvm":
        qk = ctx.kernel(d, kind, reps)
        tuned = models.tune_qsvm(qk, p.X, p.y, p.folds, ctx.grids["qsvm"], ctx.tol)
        lam = tuned.params["bandwidth"]
        K_tests = {k: qk(T, p.X, bandwidth=lam) for k, T in p.tests.items()}
        preds = models.qsvm_predict(qk(p.X, bandwidth=lam), p.y, tuned.params["C"], K_tests)
    elif model == "rbf_svm":
        tuned = models.tune_rbf(p.X, p.y, p.folds, ctx.grids["rbf_svm"], ctx.tol)
        preds = models.rbf_predict(p.X, p.y, tuned.params, p.tests)
    elif model == "isolation_forest":
        tuned = models.tune_iforest(p.X, p.y, p.folds, ctx.grids["isolation_forest"], ctx.tol, p.seed)
        preds = models.iforest_predict(p.X, p.y, tuned.params, p.seed, p.tests)
    else:
        raise ValueError(model)
    return tuned, preds


def run_models(ctx, p, names, d=None, recalls=(), **tags):
    records = []
    for name in names:
        tuned, preds = tune_and_predict(ctx, p, name, d)
        for test, (scores, pred) in preds.items():
            prevalence = test == "prevalence"
            records.append({
                "seed": p.seed, "model": name, "d": d, "test": test, **tags,
                "params": tuned.params, "cv_score": tuned.cv_score, "edges": tuned.edges,
                **metrics.evaluate(p.y_tests[test], scores, pred, recalls if prevalence else (), curve=prevalence),
            })
    return records


ALL_MODELS = ["qsvm", "rbf_svm", "isolation_forest"]
CLASSICAL = ["rbf_svm", "isolation_forest"]


def run_main(ctx):
    records = []
    for seed in ctx.seeds:
        for d in ctx.exp("main")["qubits"]:
            _log("main", "seed", seed, "d", d)
            records += run_models(ctx, ctx.prepare(seed, d), ALL_MODELS, d, features=f"pca{d}")
        records += run_models(ctx, ctx.prepare(seed), CLASSICAL, features="std30")
    ctx.save("main", records)


def run_qubits(ctx):
    records = []
    for seed in ctx.seeds:
        for d in ctx.exp("qubits")["qubits"]:
            _log("qubits", "seed", seed, "d", d)
            records += run_models(ctx, ctx.prepare(seed, d), ALL_MODELS, d, features=f"pca{d}")
    ctx.save("qubits", records)


def run_prevalence(ctx):
    cfg = ctx.exp("prevalence")
    records = []
    for seed in ctx.seeds:
        for d in cfg["qubits"]:
            _log("prevalence", "seed", seed, "d", d)
            p = ctx.prepare(seed, d, prevalence=True)
            records += run_models(ctx, p, ALL_MODELS, d, recalls=cfg["recalls"], features=f"pca{d}")
    ctx.save("prevalence", records, recall_grid=metrics.RECALL_GRID)


def run_learning_curve(ctx):
    cfg = ctx.exp("learning_curve")
    records = []
    for seed in ctx.seeds:
        for d in cfg["qubits"]:
            for n in cfg["train_sizes"]:
                _log("learning_curve", "seed", seed, "d", d, "n", n)
                records += run_models(ctx, ctx.prepare(seed, d, n_train=n), ALL_MODELS, d, n_train=n)
    ctx.save("learning_curve", records)


def run_feature_map(ctx):
    cfg = ctx.exp("feature_map")
    records = []
    for seed in ctx.seeds:
        for d in cfg["qubits"]:
            p = ctx.prepare(seed, d)
            for kind in ("z", "zz"):
                for reps in cfg["reps"]:
                    _log("feature_map", "seed", seed, "d", d, kind, "reps", reps)
                    tuned, preds = tune_and_predict(ctx, p, "qsvm", d, kind, reps)
                    scores, pred = preds["test"]
                    records.append({
                        "seed": seed, "d": d, "kind": kind, "reps": reps,
                        "params": tuned.params, "cv_score": tuned.cv_score, "edges": tuned.edges,
                        **metrics.evaluate(p.y_tests["test"], scores, pred),
                    })
    ctx.save("feature_map", records)


def run_bandwidth(ctx):
    cfg = ctx.exp("bandwidth")
    Cs = ctx.grids["wide_C"]
    records = []
    for seed in ctx.seeds:
        for d in cfg["qubits"]:
            _log("bandwidth", "seed", seed, "d", d)
            p = ctx.prepare(seed, d)
            qk = ctx.kernel(d)
            for lam in cfg["bandwidths"]:
                K = qk(p.X, bandwidth=lam)
                tuned = models.tune_C(K, p.y, p.folds, Cs, ctx.tol)
                preds = models.qsvm_predict(K, p.y, tuned.params["C"], {"test": qk(p.tests["test"], p.X, bandwidth=lam)})
                scores, pred = preds["test"]
                records.append({
                    "seed": seed, "d": d, "bandwidth": lam,
                    "params": tuned.params, "cv_score": tuned.cv_score, "edges": tuned.edges,
                    "kta": diagnostics.alignment(K, p.y), "effective_rank": diagnostics.effective_rank(K),
                    **diagnostics.concentration(K),
                    **metrics.evaluate(p.y_tests["test"], scores, pred),
                })
    ctx.save("bandwidth", records)


def run_concentration(ctx):
    cfg = ctx.exp("concentration")
    records = []
    for seed in ctx.seeds:
        for d in cfg["qubits"]:
            _log("concentration", "seed", seed, "d", d)
            p = ctx.prepare(seed, d)
            qk = ctx.kernel(d)
            for lam in cfg["bandwidths"]:
                K = qk(p.X, bandwidth=lam)
                records.append({
                    "seed": seed, "d": d, "bandwidth": lam,
                    "effective_rank": diagnostics.effective_rank(K),
                    "kta": diagnostics.alignment(K, p.y),
                    **diagnostics.concentration(K),
                })
    ctx.save("concentration", records)


def run_geometry(ctx):
    cfg = ctx.exp("geometry")
    bandwidths = ctx.grids["qsvm"]["bandwidth"]
    records, heatmap = [], {}
    for seed in ctx.seeds:
        for d in cfg["qubits"]:
            _log("geometry", "seed", seed, "d", d)
            p = ctx.prepare(seed, d)
            qk = ctx.kernel(d)
            rbf = models.tune_rbf(p.X, p.y, p.folds, ctx.grids["rbf_svm"], ctx.tol)
            q = models.tune_qsvm(qk, p.X, p.y, p.folds, ctx.grids["qsvm"], ctx.tol)
            K_rbf = rbf_kernel(p.X, gamma=rbf.params["gamma"])
            records.append({
                "seed": seed, "d": d, "kernel": "rbf", "gamma": rbf.params["gamma"],
                "kta": diagnostics.alignment(K_rbf, p.y), **diagnostics.concentration(K_rbf),
            })
            for lam in bandwidths:
                K = qk(p.X, bandwidth=lam)
                records.append({
                    "seed": seed, "d": d, "kernel": "zz", "bandwidth": lam,
                    "selected": lam == q.params["bandwidth"],
                    "kta": diagnostics.alignment(K, p.y),
                    "pearson_rbf": diagnostics.kernel_correlation(K, K_rbf),
                    "geometric_difference": diagnostics.geometric_difference(K_rbf, K, cfg["epsilon"]),
                    **diagnostics.concentration(K),
                })
            if seed == ctx.seeds[0] and d == cfg["heatmap_qubits"]:
                order = np.argsort(p.y, kind="stable")
                heatmap = {"y": p.y[order], "rbf": K_rbf[np.ix_(order, order)], "rbf_gamma": rbf.params["gamma"]}
                for lam in [*bandwidths, 1.0]:
                    heatmap[f"zz_{lam}"] = qk(p.X, bandwidth=lam)[np.ix_(order, order)]
    ctx.save("geometry", records)
    if heatmap:
        np.savez_compressed(ctx.results_dir / "geometry_heatmap.npz", **heatmap)


def run_noise(ctx):
    cfg = ctx.exp("noise")
    fm = ctx.cfg["feature_map"]
    Cs = ctx.grids["wide_C"]
    n2q = {
        d: circuit_resources(fm["kind"], d, fm["reps"], fm["entanglement"])["kernel"]["two_qubit_linear"]
        for d in {*cfg["qubits"], *cfg["unscaled_qubits"]}
    }
    settings = [("exact", None, 0.0)]
    settings += [("shots", s, 0.0) for s in cfg["shots"]]
    settings += [("depolarizing", cfg["depolarizing_shots"], e) for e in cfg["depolarizing"]]
    kernels = [("tuned", d) for d in cfg["qubits"]] + [("unscaled", d) for d in cfg["unscaled_qubits"]]

    records = []
    for seed in ctx.seeds:
        for label, d in kernels:
            _log("noise", "seed", seed, label, "d", d)
            p = ctx.prepare(seed, d)
            qk = ctx.kernel(d)
            if label == "tuned":
                lam = models.tune_qsvm(qk, p.X, p.y, p.folds, ctx.grids["qsvm"], ctx.tol).params["bandwidth"]
            else:
                lam = 1.0
            K_train, K_test = qk(p.X, bandwidth=lam), qk(p.tests["test"], p.X, bandwidth=lam)
            for i, (kind, shots, eps) in enumerate(settings):
                if label == "unscaled" and kind == "depolarizing":
                    continue
                rng = np.random.default_rng([seed, d, i])
                Ktr, Kte = noise.noisy_kernels(K_train, K_test, d, rng, shots, eps, n2q[d])
                stats = {"kta": diagnostics.alignment(Ktr, p.y), **diagnostics.concentration(Ktr)}
                Ktr, clipped, min_eig = noise.clip_psd(Ktr)
                tuned = models.tune_C(Ktr, p.y, p.folds, Cs, ctx.tol)
                scores, pred = models.qsvm_predict(Ktr, p.y, tuned.params["C"], {"test": Kte})["test"]
                records.append({
                    "seed": seed, "d": d, "kernel": label, "bandwidth": lam,
                    "noise": kind, "shots": shots, "eps": eps,
                    "survival": noise.depolarizing_survival(eps, n2q[d]), "n_two_qubit": n2q[d],
                    "psd_clipped": clipped, "min_eigenvalue": min_eig,
                    "params": tuned.params, "edges": tuned.edges, **stats,
                    **metrics.evaluate(p.y_tests["test"], scores, pred),
                })
    ctx.save("noise", records)


def run_resources(ctx):
    cfg = ctx.exp("resources")
    fm = ctx.cfg["feature_map"]
    s = ctx.cfg["sampling"]
    records = []
    for d in cfg["qubits"]:
        _log("resources", "d", d)
        records.append({"d": d, **circuit_resources(fm["kind"], d, fm["reps"], fm["entanglement"])})
    n_tr, n_te = s["n_train"], s["n_test"]
    train_pairs = n_tr * (n_tr - 1) // 2
    test_pairs = n_te * n_tr
    n_kernels = len(ctx.exp("main")["qubits"]) * len(ctx.grids["qsvm"]["bandwidth"]) * len(ctx.seeds)
    cost = {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "circuits_per_kernel": train_pairs + test_pairs,
        "shots_per_circuit": cfg["shots"],
        "shots_per_kernel": (train_pairs + test_pairs) * cfg["shots"],
        "main_kernels": n_kernels,
        "main_circuits": n_kernels * (train_pairs + test_pairs),
        "main_shots": n_kernels * (train_pairs + test_pairs) * cfg["shots"],
    }
    ctx.save("resources", records, hardware_cost=cost)


def run_pca_variance(ctx):
    qubits = ctx.exp("pca_variance")["qubits"]
    records = []
    for seed in ctx.seeds:
        split = ctx.split(seed)
        cumulative = features.pca_cumulative_variance(ctx.X[split.train(ctx.cfg["sampling"]["n_train"])], max(qubits))
        records += [{"seed": seed, "d": d, "variance": float(cumulative[d - 1])} for d in qubits]
    ctx.save("pca_variance", records)


EXPERIMENTS = {
    "pca_variance": run_pca_variance,
    "resources": run_resources,
    "main": run_main,
    "qubits": run_qubits,
    "bandwidth": run_bandwidth,
    "concentration": run_concentration,
    "geometry": run_geometry,
    "feature_map": run_feature_map,
    "learning_curve": run_learning_curve,
    "prevalence": run_prevalence,
    "noise": run_noise,
}
