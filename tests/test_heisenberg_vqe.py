
import numpy as np

from many_body_qsim.circuits import Quantum_Circuit
from many_body_qsim.lattices import square_lattice
from many_body_qsim.hamiltonians import (
    heisenberg_hamiltonian,
    pauli_basis_to_matrix
)
from many_body_qsim.algorithms import run_vqe
from many_body_qsim.ansatz import hardware_efficient_ansatz
from many_body_qsim.observables import fidelity


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
