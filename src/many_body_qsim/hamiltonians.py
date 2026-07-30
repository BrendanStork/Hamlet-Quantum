from .gates import GATES
import numpy as np

def string_to_operator(pauli_string):
    operator = GATES[pauli_string[0]] # Builds operator
    for p in pauli_string[1:]:
        operator = np.kron(operator, GATES[p])
    return operator

def pauli_basis_to_matrix(H):
    H_matrix = 0*1j
    for op, coeff in H.items():
        H_matrix += coeff * string_to_operator(op)
    return H_matrix

def general_hamiltonian(**pauli_terms): # Checks for errors then turns input into hamiltonian dict

    if not pauli_terms:
        raise ValueError('Hamiltonian cannot be empty')
        
    pauli_ops = ['I', 'X', 'Y', 'Z']
    lengths = {len(term) for term in pauli_terms} # If all the terms have the same length, set length = 1

    for term in pauli_terms:
        for char in term:
            if char not in pauli_ops:
                raise ValueError(f'Unknown operator {char}')
    
    if len(lengths) != 1:
        raise ValueError('All Pauli strings must have the same length')

    for term, coeff in pauli_terms.items():
        if not isinstance(term, str):
            raise TypeError(f'{term} is not a string')
            
        if not isinstance(coeff, (int, float, complex)):
            raise TypeError(f'Coefficient for "{term}" must be a real number')

    return pauli_terms

def ising_hamiltonian(bonds, *, J, h, transverse = True, axis = 'Z'):
    ising_hamil = {}
    sites = {site for bond in bonds for site in bond}
    site_num = max(sites) + 1

    for bond_i, bond_j in bonds:
        pauli_term = ['I'] * site_num # Initializes Pauli string to all 'I'
        pauli_term[bond_i] = 'Z'
        pauli_term[bond_j] = 'Z'
        joined_pauli_term = ''.join(pauli_term) # Joins characters into one Pauli string

        ising_hamil[joined_pauli_term] = -J
    if transverse:
        for i in range(site_num):
            pauli_term = ['I'] * site_num
            pauli_term[i] = axis
            joined_pauli_term = ''.join(pauli_term)

            ising_hamil[joined_pauli_term] = -h
    return ising_hamil


def transverse_ising_hamiltonian(bonds, *, J, h, transverse = True, axis = 'X'):
    ising_hamil = {}
    sites = {site for bond in bonds for site in bond}
    site_num = max(sites) + 1

    for bond_i, bond_j in bonds:
        pauli_term = ['I'] * site_num # Initializes Pauli string to all 'I'
        pauli_term[bond_i] = 'Z'
        pauli_term[bond_j] = 'Z'
        joined_pauli_term = ''.join(pauli_term) # Joins characters into one Pauli string

        ising_hamil[joined_pauli_term] = -J
    if transverse:
        for i in range(site_num):
            pauli_term = ['I'] * site_num
            pauli_term[i] = axis
            joined_pauli_term = ''.join(pauli_term)

            ising_hamil[joined_pauli_term] = -h
    return ising_hamil
    
def heisenberg_hamiltonian(bonds, *, Jx, Jy, Jz, h):
    
    heisen_hamil = {}
    sites = {site for bond in bonds for site in bond}
    site_num = max(sites) + 1
    
    for bond_i, bond_j in bonds:
        pauli_term = ['I'] * site_num # Initializes Pauli string to all 'I'
        pauli_term[bond_i] = 'X'
        pauli_term[bond_j] = 'X'
        joined_pauli_term = ''.join(pauli_term) # Joins characters into one Pauli string
        heisen_hamil[joined_pauli_term] = -Jx
        pauli_term = ['I'] * site_num
        pauli_term[bond_i] = 'Y'
        pauli_term[bond_j] = 'Y'
        joined_pauli_term = ''.join(pauli_term)
        heisen_hamil[joined_pauli_term] = -Jy
        pauli_term = ['I'] * site_num
        pauli_term[bond_i] = 'Z'
        pauli_term[bond_j] = 'Z'
        joined_pauli_term = ''.join(pauli_term)
        heisen_hamil[joined_pauli_term] = -Jz
        
    for i in range(site_num):
        pauli_term = ['I'] * site_num
        pauli_term[i] = 'Z'
        joined_pauli_term = ''.join(pauli_term)
        heisen_hamil[joined_pauli_term] = -h
        
    return heisen_hamil


def hubbard_hamiltonian(bonds, *, t, U):
    """
    Constructs the Fermi-Hubbard Hamiltonian in the Jordan-Wigner
    Pauli-string representation.

    Parameters
    ----------
    bonds : list of tuples
        Lattice bonds, e.g. [(0,1), (1,2)]

    t : float
        Hopping strength

    U : float
        Onsite interaction strength

    Returns
    -------
    hamiltonian : dict
        Dictionary mapping Pauli strings -> coefficients
    """

    hubbard_hamil = {}

    # ---------------------------------------------------
    # Determine number of physical sites
    # ---------------------------------------------------

    sites = {site for bond in bonds for site in bond}
    num_sites = max(sites) + 1

    # Total qubits = 2 spin orbitals per site
    num_qubits = 2 * num_sites

    # ---------------------------------------------------
    # Helper function:
    # add Pauli term to dictionary
    # ---------------------------------------------------

    def add_term(pauli_string, coeff):

        if pauli_string in hubbard_hamil:
            hubbard_hamil[pauli_string] += coeff
        else:
            hubbard_hamil[pauli_string] = coeff

    # ===================================================
    # HOPPING TERMS
    # ===================================================

    for site_i, site_j in bonds:

        # Ensure ordered indices for JW strings
        i, j = sorted((site_i, site_j))

        # ------------------------------------------------
        # SPIN-UP HOPPING
        # modes: 2i <-> 2j
        # ------------------------------------------------

        up_i = 2 * i
        up_j = 2 * j

        # ----- XZX term -----

        pauli = ['I'] * num_qubits

        pauli[up_i] = 'X'
        pauli[up_j] = 'X'

        for k in range(up_i + 1, up_j):
            pauli[k] = 'Z'

        add_term(''.join(pauli), t / 2)

        # ----- YZY term -----

        pauli = ['I'] * num_qubits

        pauli[up_i] = 'Y'
        pauli[up_j] = 'Y'

        for k in range(up_i + 1, up_j):
            pauli[k] = 'Z'

        add_term(''.join(pauli), t / 2)

        # ------------------------------------------------
        # SPIN-DOWN HOPPING
        # modes: 2i+1 <-> 2j+1
        # ------------------------------------------------

        down_i = 2 * i + 1
        down_j = 2 * j + 1

        # ----- XZX term -----

        pauli = ['I'] * num_qubits

        pauli[down_i] = 'X'
        pauli[down_j] = 'X'

        for k in range(down_i + 1, down_j):
            pauli[k] = 'Z'

        add_term(''.join(pauli), t / 2)

        # ----- YZY term -----

        pauli = ['I'] * num_qubits

        pauli[down_i] = 'Y'
        pauli[down_j] = 'Y'

        for k in range(down_i + 1, down_j):
            pauli[k] = 'Z'

        add_term(''.join(pauli), t / 2)

    # ===================================================
    # ONSITE INTERACTION TERMS
    # ===================================================

    for i in range(num_sites):

        up = 2 * i
        down = 2 * i + 1

        # -----------------------------------------------
        # + (U/4) I
        # -----------------------------------------------

        add_term('I' * num_qubits, U / 4)

        # -----------------------------------------------
        # - (U/4) Z_up
        # -----------------------------------------------

        pauli = ['I'] * num_qubits
        pauli[up] = 'Z'

        add_term(''.join(pauli), -U / 4)

        # -----------------------------------------------
        # - (U/4) Z_down
        # -----------------------------------------------

        pauli = ['I'] * num_qubits
        pauli[down] = 'Z'

        add_term(''.join(pauli), -U / 4)

        # -----------------------------------------------
        # + (U/4) Z_up Z_down
        # -----------------------------------------------

        pauli = ['I'] * num_qubits

        pauli[up] = 'Z'
        pauli[down] = 'Z'

        add_term(''.join(pauli), U / 4)

    return hubbard_hamil


def hubbard_spectrum(bonds, * , t, U, num_fermions):
    sites = {site for bond in bonds for site in bond}
    num_sites = max(sites) + 1 #
    H_pauli = hubbard_hamiltonian(bonds, t=t, U=U)
    dimension = 4**num_sites # Hilbert dimension for fermions
    N_list = [bin(i).count("1") for i in range(dimension)] # For every basis state, lists count of fermions
    N_target = num_fermions # Desired particle number subspace target (e.g. N = 2 fermions in system)
    
    H = 0*1j
    for op, coeff in basis.items():
        H += coeff * string_to_operator(op)

    indices = [i for i in range(16) if N_list[i] == N_target] # Which basis state indices have N_target fermions
    H_N = H[np.ix_(indices, indices)] # Create Hamiltonian subpace for N_target
    E_N = np.linalg.eigh(H_N) # Energy spectrum for H_N
    return E_N
