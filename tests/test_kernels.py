import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qkfraud import noise, stats
from qkfraud.quantum import QuantumKernel, circuit_resources, feature_map, kernel_circuit


@pytest.fixture
def X():
    return np.random.default_rng(0).uniform(0, np.pi, size=(12, 3))


@pytest.mark.parametrize("reps", [1, 2, 3])
def test_z_map_closed_form_matches_statevector(X, reps):
    qc = feature_map("z", 3, reps)
    states = np.stack([Statevector(qc.assign_parameters(x)).data for x in X])
    expected = np.abs(states @ states.conj().T) ** 2
    assert np.allclose(QuantumKernel("z", 3, reps)(X), expected)


def test_zz_kernel_is_valid_gram(X):
    K = QuantumKernel("zz", 3)(X, bandwidth=0.5)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_kernel_equals_compute_uncompute_probability(X):
    qc = kernel_circuit("zz", 3).remove_final_measurements(inplace=False)
    bound = qc.assign_parameters(np.concatenate([X[0], X[1]]))
    prob = abs(Statevector(bound).data[0]) ** 2
    assert np.isclose(prob, QuantumKernel("zz", 3)(X[:1], X[1:2])[0, 0])


def test_resources_kernel_doubles_feature_map():
    r = circuit_resources("zz", 4, 2)
    assert r["kernel"]["two_qubit"] == 2 * r["feature_map"]["two_qubit"] == 24


def test_shot_noise_keeps_symmetry():
    K = QuantumKernel("zz", 3)(np.random.default_rng(1).uniform(0, 1, (8, 3)))
    Kn = noise.sample_shots(K, 100, np.random.default_rng(0), train=True)
    assert np.allclose(Kn, Kn.T) and np.allclose(np.diag(Kn), 1)


def test_holm_is_monotone():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
