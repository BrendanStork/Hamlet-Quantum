
import numpy as np

from many_body_qsim.lattices import square_lattice
from many_body_qsim.hamiltonians import (
    ising_hamiltonian,
    pauli_basis_to_matrix
)
from many_body_qsim.algorithms import run_qaoa


def test_ising_qaoa_workflow():

    # Build lattice
    Nx = 2
    Ny = 2
    N = Nx * Ny

    bonds = square_lattice(
        Nx,
        Ny
    )


    # Ising cost Hamiltonian
    H = ising_hamiltonian(
        bonds,
        J=1,
        h=1,
        axis='Z'
    )


    # Exact ground energy

    H_matrix = pauli_basis_to_matrix(H)

    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

    exact_energy = eigenvalues[0]


    # Run QAOA

    result = run_qaoa(
        H,
        p=2,
        optimizer='COBYLA'
    )


    qaoa_energy = result['cost']


    # Validate

    assert np.isfinite(qaoa_energy)

    assert qaoa_energy <= exact_energy + 1e-2
