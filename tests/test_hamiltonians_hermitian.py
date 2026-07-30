
import numpy as np

from many_body_qsim.lattices import square_lattice
from many_body_qsim.hamiltonians import (
    string_to_operator,
    general_hamiltonian,
    transverse_ising_hamiltonian,
    heisenberg_hamiltonian,
    hubbard_hamiltonian
    )

bonds = square_lattice(2,2)

def test_general_is_hermitian():

    H_pauli = general_hamiltonian(XYX = 3.0, ZZY = .4, ZZZ = 9)
    H = 0*1j
    for op, coeff in H_pauli.items():
        H += coeff * string_to_operator(op)
        
    np.testing.assert_allclose(
        H,
        H.conj().T
    )

def test_tfim_is_hermitian():

    H_pauli = transverse_ising_hamiltonian(bonds, J=1.0, h=0.5)
    H = 0*1j
    for op, coeff in H_pauli.items():
        H += coeff * string_to_operator(op)
        
    np.testing.assert_allclose(
        H,
        H.conj().T
    )
    
def test_heisen_is_hermitian():

    H_pauli = heisenberg_hamiltonian(bonds, Jx=4.0, Jy=.5, Jz = 10, h = 7.2)
    H = 0*1j
    for op, coeff in H_pauli.items():
        H += coeff * string_to_operator(op)
        
    np.testing.assert_allclose(
        H,
        H.conj().T
    )
    
def test_hubbard_is_hermitian():

    H_pauli = hubbard_hamiltonian(bonds, t=5.2, U=7.7)
    H = 0*1j
    for op, coeff in H_pauli.items():
        H += coeff * string_to_operator(op)
        
    np.testing.assert_allclose(
        H,
        H.conj().T
    )
