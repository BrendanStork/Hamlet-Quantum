
# tests/test_qft.py

import numpy as np
import pytest

from hamlet.circuits import Quantum_Circuit
from hamlet.algorithms import qft, inverse_qft   # adjust import path



ATOL = 1e-10


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def basis_state(n, index):
    """Return |index> for an n-qubit big-endian statevector."""
    state = np.zeros(2**n, dtype=complex)
    state[index] = 1.0
    return state


def random_state(n, seed=1234):
    """Return a normalized reproducible random complex state."""
    rng = np.random.default_rng(seed)

    state = (
        rng.normal(size=2**n)
        + 1j * rng.normal(size=2**n)
    )

    return state / np.linalg.norm(state)


def qft_matrix(n):
    """
    Exact mathematical QFT using the convention

        F[j, k] = exp(+2π i j k / N) / sqrt(N)
    """
    N = 2**n

    row = np.arange(N).reshape(N, 1)
    col = np.arange(N).reshape(1, N)

    return (
        np.exp(2j * np.pi * row * col / N)
        / np.sqrt(N)
    )


# ------------------------------------------------------------
# Reference matrix sanity checks
# ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_reference_matrix_is_unitary(n):
    F = qft_matrix(n)
    identity = np.eye(2**n)

    assert np.allclose(
        F.conj().T @ F,
        identity,
        atol=ATOL,
    )


# ------------------------------------------------------------
# Simple known-state tests
# ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_zero_state_gives_uniform_superposition(n):
    qc = Quantum_Circuit(n)

    # Assuming circuits initialize in |00...0>
    qft(qc, list(range(n)))

    expected = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    assert np.allclose(
        qc.state,
        expected,
        atol=ATOL,
    )


def test_qft_one_state_three_qubits():
    n = 3

    qc = Quantum_Circuit(n)
    qc.state = basis_state(n, 1)  # |001>

    qft(qc, [0, 1, 2])

    expected = qft_matrix(n) @ basis_state(n, 1)

    assert np.allclose(
        qc.state,
        expected,
        atol=ATOL,
    )


# ------------------------------------------------------------
# Exhaustive computational-basis tests
# ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_all_basis_states_against_exact_matrix(n):
    F = qft_matrix(n)

    for x in range(2**n):
        initial = basis_state(n, x)
        expected = F @ initial

        qc = Quantum_Circuit(n)
        qc.state = initial.copy()

        qft(qc, list(range(n)))

        assert np.allclose(
            qc.state,
            expected,
            atol=ATOL,
        ), f"QFT failed for n={n}, basis state |{x}>"



@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_inverse_qft_all_basis_states_against_exact_matrix(n):
    F_inv = qft_matrix(n).conj().T

    for x in range(2**n):
        initial = basis_state(n, x)
        expected = F_inv @ initial

        qc = Quantum_Circuit(n)
        qc.state = initial.copy()

        inverse_qft(qc, list(range(n)))

        assert np.allclose(
            qc.state,
            expected,
            atol=ATOL,
        ), f"inverse QFT failed for n={n}, basis state |{x}>"



# ------------------------------------------------------------
# Arbitrary-state tests
# ------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_qft_random_state_against_exact_matrix(n):
    initial = random_state(n, seed=100 + n)

    expected = qft_matrix(n) @ initial

    qc = Quantum_Circuit(n)
    qc.state = initial.copy()

    qft(qc, list(range(n)))

    assert np.allclose(
        qc.state,
        expected,
        atol=ATOL,
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_inverse_qft_random_state_against_exact_matrix(n):
    initial = random_state(n, seed=200 + n)

    expected = qft_matrix(n).conj().T @ initial

    qc = Quantum_Circuit(n)
    qc.state = initial.copy()

    inverse_qft(qc, list(range(n)))

    assert np.allclose(
        qc.state,
        expected,
        atol=ATOL,
    )


# ------------------------------------------------------------
# QFT / inverse-QFT round trip
# ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_qft_inverse_qft_roundtrip(n):
    initial = random_state(n, seed=300 + n)

    qc = Quantum_Circuit(n)
    qc.state = initial.copy()

    qft(qc, list(range(n)))
    inverse_qft(qc, list(range(n)))

    assert np.allclose(
        qc.state,
        initial,
        atol=ATOL,
    )


# ------------------------------------------------------------
# Norm preservation
# ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_preserves_norm(n):
    initial = random_state(n, seed=400 + n)

    qc = Quantum_Circuit(n)
    qc.state = initial.copy()

    qft(qc, list(range(n)))

    assert np.isclose(
        np.linalg.norm(qc.state),
        1.0,
        atol=ATOL,
    )


# ------------------------------------------------------------
# Swap handling
# ------------------------------------------------------------

def test_qft_swaps_change_only_output_order():
    """
    For 3 qubits, swaps=True should be equivalent to
    swaps=False followed by SWAP(0, 2).
    """
    n = 3
    initial = random_state(n, seed=500)

    qc_no_swap = Quantum_Circuit(n)
    qc_no_swap.state = initial.copy()

    qft(qc_no_swap, [0, 1, 2], swaps=False)

    qc_no_swap.swap(0, 2)

    qc_swap = Quantum_Circuit(n)
    qc_swap.state = initial.copy()

    qft(qc_swap, [0, 1, 2], swaps=True)

    assert np.allclose(
        qc_no_swap.state,
        qc_swap.state,
        atol=ATOL,
    )


def test_qft_swaps_four_qubits():
    """
    Explicitly test both reversal swaps:

        q0 <-> q3
        q1 <-> q2
    """
    n = 4
    initial = random_state(n, seed=501)

    qc_no_swap = Quantum_Circuit(n)
    qc_no_swap.state = initial.copy()

    qft(qc_no_swap, [0, 1, 2, 3], swaps=False)

    qc_no_swap.swap(0, 3)
    qc_no_swap.swap(1, 2)

    qc_swap = Quantum_Circuit(n)
    qc_swap.state = initial.copy()

    qft(qc_swap, [0, 1, 2, 3], swaps=True)

    assert np.allclose(
        qc_no_swap.state,
        qc_swap.state,
        atol=ATOL,
    )


# ------------------------------------------------------------
# Subregister test -- important for QPE
# ------------------------------------------------------------

def test_qft_only_acts_on_selected_ancilla_qubits():
    """
    3 ancilla qubits + 1 system qubit.

    QFT should act on q0,q1,q2 and leave q3 unchanged.
    """

    n_ancilla = 3

    ancilla = random_state(n_ancilla, seed=600)

    # Put the system qubit in |1>
    system = np.array([0.0, 1.0], dtype=complex)

    # Big-endian ordering:
    # |ancilla> tensor |system>
    initial = np.kron(ancilla, system)

    expected_ancilla = qft_matrix(n_ancilla) @ ancilla
    expected = np.kron(expected_ancilla, system)

    qc = Quantum_Circuit(4)
    qc.state = initial.copy()

    qft(qc, [0, 1, 2])

    assert np.allclose(
        qc.state,
        expected,
        atol=ATOL,
    )
