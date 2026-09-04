
import numpy as np

from hamlet.circuits import Quantum_Circuit
from hamlet.lattices import square_lattice
from hamlet.hamiltonians import (
    heisenberg_hamiltonian,
    pauli_basis_to_matrix
)
from hamlet.algorithms import run_vqe
from hamlet.ansatz import hardware_efficient_ansatz
from hamlet.observables import fidelity


def test_heisenberg_vqe_workflow():

    # Build lattice
    Nx = 2
    Ny = 2
    N = Nx * Ny

    bonds = square_lattice(Nx, Ny)

    # Build Hamiltonian
    H = heisenberg_hamiltonian(
        bonds,
        Jx=-1,
        Jy=-1,
        Jz=-1,
        h=0
    )

    # Exact solution
    H_matrix = pauli_basis_to_matrix(H)

    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

    exact_energy = eigenvalues[0]
    exact_state = eigenvectors[:, 0]


    # Run VQE
    layers = 3

    result = run_vqe(
        H,
        ansatz=hardware_efficient_ansatz,
        method='L-BFGS-B',
        layers=layers
    )


    vqe_energy = result.fun
    params = result.x


    # Construct optimized state

    qc = Quantum_Circuit(N)

    vqe_state = hardware_efficient_ansatz(
        qc,
        params,
        layers=layers
    )


    # Energy should be close
    assert abs(vqe_energy - exact_energy) < 1e-2
