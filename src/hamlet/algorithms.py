import numpy as np
from .circuits import Quantum_Circuit
from .gates import apply_hadamard_all
from scipy.optimize import minimize
 
'''
VQE
'''


def vqe_energy(
    init_params,
    H,
    ansatz,
    layers=1):
    
    '''
    The number of qubits in this iteration is half the number of initial parameters 
    divided by the number of layers
    '''
    
    num_qubits = int(len(init_params)/2/layers)
    qc = Quantum_Circuit(num_qubits)
    
    ansatz(
        qc,
        init_params,
        layers=layers
    )

    energy = 0


    for pauli, coefficient in H.items():

        expectation = qc.expectation_value(pauli)
        energy += coefficient * expectation

    return energy
    
    
def run_vqe(
    H,
    *,
    ansatz,
    method,
    layers=1):

    num_parameters = 2 * len(next(iter(H))) * layers
    init_params = np.random.uniform(0, 2*np.pi, size=num_parameters)

    result = minimize(
        vqe_energy,
        init_params,
        args=(
            H,
            ansatz,
            layers
        ),
        method=method,
        options={'maxiter':10000}
    )


    return result

'''
QAOA
'''


def mixer_hamiltonian(n_qubits):
    mixer_H = {}
    for i in range(n_qubits):
        pauli_term = ['I'] * n_qubits
        pauli_term[i] = 'X'
        joined_pauli_term = ''.join(pauli_term)
        mixer_H[joined_pauli_term] = 1
    return mixer_H

def pauli_exponential_evolution(qc, H, angle):

    length_H_term = len(next(iter(H))) # Grabs the first basis state's length
    num_qubits = qc.numqubits

    if num_qubits != length_H_term:
        raise ValueError('Length of Pauli strings must equal number of qubits')
        
    for pauli_string, coeff in H.items():

        active_qubits = []
        
        # -------------------------
        # 1. BASIS ROTATIONS
        # -------------------------
        
        for q in range(num_qubits):

            p = pauli_string[q]

            if p != 'I':
                active_qubits.append(q)
            
            if p == 'X':
                qc.h(q)

                qc.gate_count += 1
            elif p == 'Y':
                qc.sdag(q)
                qc.h(q)
                qc.gate_count += 2

        # -------------------------
        # 2. ENTANGLE PARITY
        # -------------------------

        for i in range(len(active_qubits) - 1):

            qc.cx(active_qubits[i],
                  active_qubits[i + 1])
            qc.gate_count += 1

        # -------------------------
        # 3. PHASE ROTATION
        # -------------------------
        if active_qubits:
            qc.rz(
                active_qubits[-1],
                2 * coeff * angle
            )
            qc.gate_count += 1

        # -------------------------
        # 4. UNCOMPUTE PARITY
        # -------------------------
        for i in reversed(range(len(active_qubits) - 1)):
            qc.cx(active_qubits[i],
                  active_qubits[i + 1])
            qc.gate_count += 1

        # -------------------------
        # 5. UNDO BASIS ROTATIONS
        # -------------------------
        for q in range(num_qubits):

            p = pauli_string[q]

            if p == 'X':
                qc.h(q)
                qc.gate_count += 1
            elif p == 'Y':
                qc.h(q)
                qc.s(q)
                qc.gate_count += 2

    return qc
    
    

def qaoa_state(
    params,
    cost_hamiltonian,
    n_qubits,
    p
):

    gamma = params[:p]

    beta = params[p:]

    qc = Quantum_Circuit(n_qubits)

    apply_hadamard_all(qc)

    mixer_H = mixer_hamiltonian(n_qubits)
    cost_H = cost_hamiltonian

    for i in range(p):
        pauli_exponential_evolution(
            qc,
            cost_H,
            gamma[i]
        )

        pauli_exponential_evolution(
            qc,
            mixer_H,
            beta[i]
        )

    return qc

def qaoa_energy(
    params,
    cost_hamiltonian,
    n_qubits,
    p
):
    
    qc = qaoa_state(
        params,
        cost_hamiltonian,
        n_qubits,
        p
    )
    
    
    energy = 0
    
    for pauli, coefficient in cost_hamiltonian.items():

        expectation = qc.expectation_value(pauli)
        energy += coefficient * expectation
    
    
    return energy
    
def run_qaoa(
    cost_hamiltonian,
    p=1,
    optimizer='COBYLA'
):

    n_qubits = len(next(iter(cost_hamiltonian)))
    initial_params = np.random.uniform(
        0,
        2*np.pi,
        2*p
    )

    

    result = minimize(
        qaoa_energy,
        initial_params,
        args=(
            cost_hamiltonian,
            n_qubits,
            p
        ),
        method=optimizer,
        options={'maxiter':10000}
    )


    final_state = qaoa_state(
        result.x,
        cost_hamiltonian,
        n_qubits,
        p
    )


    #return result
    
    return {
        'result': result,
        'parameters': result.x,
        'state': final_state,
        'cost': result.fun
    }
    
    
def qaoa_energy_map(
    cost_hamiltonian,
    *,
    gammas,
    betas
):

    n_qubits = len(next(iter(cost_hamiltonian)))
    energy_map = np.zeros((len(gammas), len(betas)))
    params = []
    p = 1
    
    for i in range(len(gammas)):
        for j in range(len(betas)):
            params = [gammas[i], betas[j]]
            energy_map[i, j] = qaoa_energy(
                params,
                cost_hamiltonian,
                n_qubits,
                p
            )

    return energy_map

def p_val_probability(cost_hamiltonian, p):
    n_qubits = len(next(iter(cost_hamiltonian)))
    prob_array = np.zeros((4, 2**n_qubits))
    for i in range(1, 1+p):
    
        result = run_qaoa(cost_hamiltonian, p=i)
        
        prob_array[i-1, :] = np.abs(result['state'].state)**2
    return prob_array

def enumerate_probabilities(qc):
    probabilities = np.abs(qc.state)**2
    
    for state, probability in enumerate(probabilities):
        print(
            format(state, f'0{N}b'),
            probability
        )
