import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

try:
    from qiskit.circuit.library import z_feature_map, zz_feature_map
except ImportError:  # Qiskit < 2.1
    from qiskit.circuit.library import ZFeatureMap as z_feature_map
    from qiskit.circuit.library import ZZFeatureMap as zz_feature_map

LOGICAL_BASIS = ["h", "p", "cx"]
HARDWARE_BASIS = ["rz", "sx", "x", "cx"]


def feature_map(kind, d, reps=2, entanglement="linear", prefix="x"):
    if kind == "zz":
        return zz_feature_map(feature_dimension=d, reps=reps, entanglement=entanglement, parameter_prefix=prefix)
    if kind == "z":
        return z_feature_map(feature_dimension=d, reps=reps, parameter_prefix=prefix)
    raise ValueError(f"unknown feature map {kind!r}")


def kernel_circuit(kind, d, reps=2, entanglement="linear"):
    """Compute-uncompute circuit U(x')^dagger U(x); k(x, x') is P(all zeros)."""
    u = feature_map(kind, d, reps, entanglement, "x")
    v = feature_map(kind, d, reps, entanglement, "y")
    qc = u.compose(v.inverse())
    qc.measure_all()
    return qc


def _depth_and_cx(qc):
    depth = qc.depth(lambda inst: inst.operation.name not in ("measure", "barrier"))
    return depth, int(qc.count_ops().get("cx", 0))


def circuit_resources(kind, d, reps=2, entanglement="linear"):
    """Depth and two-qubit gate count, logical and after transpiling to a linear device."""
    out = {}
    for name, qc in (
        ("feature_map", feature_map(kind, d, reps, entanglement)),
        ("kernel", kernel_circuit(kind, d, reps, entanglement)),
    ):
        logical = transpile(qc, basis_gates=LOGICAL_BASIS, optimization_level=0)
        linear = transpile(
            qc,
            basis_gates=HARDWARE_BASIS,
            coupling_map=CouplingMap.from_line(d),
            initial_layout=list(range(d)),
            optimization_level=1,
            seed_transpiler=0,
        )
        depth, cx = _depth_and_cx(logical)
        depth_lin, cx_lin = _depth_and_cx(linear)
        out[name] = {"depth": depth, "depth_linear": depth_lin, "two_qubit": cx, "two_qubit_linear": cx_lin}
    return out


def single_qubit_states(X, reps):
    """Product states of the Z feature map, shape (n, d, 2): (P(2x) H)^reps |0>."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    psi = np.zeros(X.shape + (2,), dtype=complex)
    psi[..., 0] = 1.0
    for _ in range(reps):
        psi = psi @ h.T
        psi[..., 1] *= np.exp(2j * X)
    return psi


def check_gram(K, eig_tol=1e-8):
    if not np.allclose(K, K.T, atol=1e-10):
        raise ValueError("kernel matrix is not symmetric")
    if not np.allclose(np.diag(K), 1.0, atol=1e-10):
        raise ValueError("kernel matrix does not have a unit diagonal")
    if K.min() < -1e-10 or K.max() > 1.0 + 1e-10:
        raise ValueError("kernel entries outside [0, 1]")
    if np.linalg.eigvalsh(K).min() < -eig_tol:
        raise ValueError("kernel matrix is not positive semi-definite")
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K


class KernelCache:
    def __init__(self, root, max_entries=2_000_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    @staticmethod
    def key(meta, A, B):
        h = hashlib.sha1(json.dumps(meta, sort_keys=True).encode())
        h.update(np.ascontiguousarray(A, dtype=float).tobytes())
        h.update(b"sym" if B is None else np.ascontiguousarray(B, dtype=float).tobytes())
        return h.hexdigest()

    def load(self, key):
        path = self.root / f"{key}.npy"
        return np.load(path) if path.exists() else None

    def save(self, key, K):
        if K.size <= self.max_entries:
            np.save(self.root / f"{key}.npy", K)


class QuantumKernel:
    """Exact fidelity kernel k(x, x') = |<phi(x')|phi(x)>|^2 with bandwidth scaling x -> lambda x."""

    def __init__(self, kind="zz", d=4, reps=2, entanglement="linear", cache=None):
        self.kind, self.d, self.reps, self.entanglement = kind, d, reps, entanglement
        self.cache = cache
        self._circuit = feature_map(kind, d, reps, entanglement) if kind == "zz" else None

    @property
    def meta(self):
        return {"kind": self.kind, "d": self.d, "reps": self.reps, "entanglement": self.entanglement}

    def states(self, X):
        if self.kind == "z":
            return single_qubit_states(X, self.reps)
        return np.stack([Statevector(self._circuit.assign_parameters(x)).data for x in X])

    def _evaluate(self, A, B):
        SA = self.states(A)
        SB = SA if B is None else self.states(B)
        if self.kind == "z":
            K = np.ones((len(SA), len(SB)))
            for j in range(self.d):
                K *= np.abs(SA[:, j] @ SB[:, j].conj().T) ** 2
            return K
        return np.abs(SA @ SB.conj().T) ** 2

    def __call__(self, A, B=None, bandwidth=1.0):
        key = None
        if self.cache is not None:
            key = self.cache.key({**self.meta, "bandwidth": float(bandwidth)}, A, B)
            K = self.cache.load(key)
            if K is not None:
                return K
        K = self._evaluate(bandwidth * A, None if B is None else bandwidth * B)
        if B is None:
            K = check_gram(K)
        if key is not None:
            self.cache.save(key, K)
        return K
