import numpy as np
from .circuits import Quantum_Circuit
from scipy import linalg
from .gates import GATES


def string_to_operator(pauli_string):
    operator = GATES[pauli_string[0]] # Builds operator
    for p in pauli_string[1:]:
        operator = np.kron(operator, GATES[p])
    return operator

def exact_evolve(qc0, basis, time):
    H = 0*1j
    for op, coeff in basis.items():
        H += coeff * string_to_operator(op)
    U = linalg.expm(-1j*H*time)
    new_state = qc0
    new_state.state = U @ qc0.state
    return new_state

######GENERAL N QUBIT TROTTER#########


def trotter_evolve(qc0, basis, time=1, trotter_steps=1):
    qc = qc0
    length_basis_term = len(next(iter(basis))) # Grabs the first basis state's length
    num_qubits = int(np.log2(len(qc.state))) # Gives number of qubits based on state coefficient length

    if num_qubits != length_basis_term:
        raise ValueError('Length of Pauli strings must equal number of qubits')
    
    dt = time / trotter_steps

    for _ in range(trotter_steps):

        for pauli_string, coeff in basis.items():
            active_qubits = []
            
            # Basis rotations
            
            for q in range(num_qubits):

                p = pauli_string[q]
                if p != 'I':
                    active_qubits.append(q)
                
                if p == 'X':
                    qc.h(q)

                elif p == 'Y':
                    qc.sdag(q)
                    qc.h(q)

            # Entangle parity (apply CNOT chain)

            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i],
                      active_qubits[i + 1])


            # Phase rotation
            
            if active_qubits:
                qc.rz(
                    active_qubits[-1],
                    2 * coeff * dt
                )

            # Uncompute parity (undo CNOT chain)
            
            for i in reversed(range(len(active_qubits) - 1)):
                qc.cx(active_qubits[i],
                      active_qubits[i + 1])

            # Undo basis rotations
            
            for q in range(num_qubits):

                p = pauli_string[q]

                if p == 'X':
                    qc.h(q)
                
                elif p == 'Y':
                    qc.h(q)
                    qc.s(q)

    return qc


def trotter_step(qc, basis, *, dt):

    length_basis_term = len(next(iter(basis))) # Grabs the first basis state's length
    num_qubits = qc.numqubits
    
    if num_qubits != length_basis_term:
        raise ValueError('Length of Pauli strings must equal number of qubits')
        
    for pauli_string, coeff in basis.items():
            #print(pauli_string)
            active_qubits = []
            
            # -------------------------
            # 1. BASIS ROTATIONS
            # -------------------------
            
            for q in range(num_qubits):

                p = pauli_string[q]
                #print(p, q)
                if p != 'I':
                    active_qubits.append(q)
                
                if p == 'X':
                    qc.h(q)
                    #print('H', q)
                    qc.gate_count += 1
                elif p == 'Y':
                    qc.sdag(q)
                    qc.h(q)
                    qc.gate_count += 2
                    #print('Sdag', q)
                    #print('H', q)
            #print(active_qubits)
            # -------------------------
            # 2. ENTANGLE PARITY
            # -------------------------
            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i],
                      active_qubits[i + 1])
                qc.gate_count += 1
                #print(active_qubits[i],
                 #         active_qubits[i + 1])
            # -------------------------
            # 3. PHASE ROTATION
            # -------------------------
            if active_qubits:
                qc.rz(
                    active_qubits[-1],
                    2 * coeff * dt
                )
                qc.gate_count += 1
                #print('Rz', active_qubits[-1], coeff)
            # -------------------------
            # 4. UNCOMPUTE PARITY
            # -------------------------
            for i in reversed(range(len(active_qubits) - 1)):
                qc.cx(active_qubits[i],
                      active_qubits[i + 1])
                qc.gate_count += 1
                #print(active_qubits[i],
                 #     active_qubits[i + 1])
            # -------------------------
            # 5. UNDO BASIS ROTATIONS
            # -------------------------
            for q in range(num_qubits):

                p = pauli_string[q]

                if p == 'X':
                    qc.h(q)
                    #print('H', q)
                    qc.gate_count += 1
                elif p == 'Y':
                    qc.h(q)
                    qc.s(q)
                    qc.gate_count += 2
                    #print('H', q)
                    #print('S', q)
    return qc
    
