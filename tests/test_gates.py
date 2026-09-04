import numpy as np
import pytest
from hamlet.gates import GATES, apply_cnot, apply_hadamard_all
from hamlet.circuits import Quantum_Circuit

theta = np.random.default_rng().uniform(0,2*np.pi)

@pytest.mark.parametrize(
'gate',
['X', 'Y', 'Z', 'H', 'S', 'Sdag', 'T']
)
def test_single_gate_unitary(gate):
    U = GATES[gate]
    I = np.eye(2)
    np.testing.assert_allclose(U.conj().T @ U, I, atol=1e-7)


@pytest.mark.parametrize(
'gate',
['RX', 'RY', 'RZ']
)
def test_rotation_gate_unitary(gate):
    U = GATES[gate](theta)
    I = np.eye(2)
    np.testing.assert_allclose(U.conj().T @ U, I, atol=1e-7)


def test_two_qubit_gate():
    qc0 = Quantum_Circuit(2)
    qc0.x(0)
    qc_state = apply_cnot(qc0.state, 0, 1)
    np.testing.assert_allclose(qc_state, np.array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]), atol=1e-7)
