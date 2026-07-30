import numpy as np
from .gates import GATES, apply_cnot


def build_full_operator(gate_matrix, target_qubit, num_qubits):
    if target_qubit >= num_qubits:
        raise ValueError("Invalid qubit index")

    ops = []

    for i in range(num_qubits):
        if i == target_qubit:
            ops.append(gate_matrix)     # already a matrix
        else:
            ops.append(GATES['I'])      # matrix

    full_op = ops[0]

    for op in ops[1:]:
        full_op = np.kron(full_op, op)

    return full_op


class Quantum_Circuit:
    def __init__(self, numqubits):

        self.numqubits = numqubits
        self.state = np.zeros(2**numqubits, dtype=complex)
        self.state[0] = 1
        self.gate_count = 0

    def gate_op(self, gate, target):

        bit = self.numqubits - 1 - target # Index change from big endian to little endian
    
        for i in range(len(self.state)):
    
            # Only processes "0" side of pair
            if ((i >> bit) & 1) == 0:
    
                j = (i | (1 << bit)) # Finds pair base
    
                a0 = self.state[i]
                a1 = self.state[j]
    
                self.state[i] = (
                    gate[0,0] * a0 + gate[0,1] * a1
                )
                self.state[j] = (
                    gate[1,0] * a0 + gate[1,1] * a1
                )

        return self.state
    

    def x(self, qubitIndex):
        self.gate_op(GATES['X'], qubitIndex)
        return

        
    def y(self, qubitIndex):
        self.gate_op(GATES['Y'], qubitIndex)
        return
        
    def z(self, qubitIndex):
        self.gate_op(GATES['Z'], qubitIndex)
        return
        
    def h(self, qubitIndex):
        self.gate_op(GATES['H'], qubitIndex)
        return

    def s(self, qubitIndex):
        self.gate_op(GATES['S'], qubitIndex)
        return

    def t(self, qubitIndex):
        self.gate_op(GATES['T'], qubitIndex)
        return

    def sdag(self, qubitIndex):
        self.gate_op(GATES['Sdag'], qubitIndex)
        return

    def p(self, qubitIndex, theta):
        self.gate_op(GATES['P'](theta), qubitIndex)
        return

    def rx(self, qubitIndex, theta):
        self.gate_op(GATES['RX'](theta), qubitIndex)
        return

    def ry(self, qubitIndex, theta):
        self.gate_op(GATES['RY'](theta), qubitIndex)
        return
        
    def rz(self, qubitIndex, theta):
        self.gate_op(GATES['RZ'](theta), qubitIndex)
        return

    def cx(self, control, target):
        self.state = apply_cnot(self.state, control, target)
        return
        
    '''
    def expectation_value(self, pauli_string):

        if len(pauli_string) != self.numqubits:
            raise ValueError('Pauli string length must equal number of qubits')

        operator = GATES[pauli_string[0]] # Builds operator
        for p in pauli_string[1:]:
            operator = np.kron(operator, GATES[p])
        return np.vdot(self.state, operator @ self.state).real # Expectation value
    
    '''
    def expectation_value(self, pauli_string):

        if len(pauli_string) != self.numqubits:
            raise ValueError('Pauli string length must equal number of qubits')
            
        expectation = 0.0

        for i in range(len(self.state)):

            j = i
            phase = 1.0 + 0j

            for q, p in enumerate(pauli_string):

                bit = self.numqubits - 1 - q
                bit_value = (i >> bit) & 1

                if p == 'X':
                    j ^= (1 << bit)

                elif p == 'Y':
                    j ^= (1 << bit)

                    if bit_value == 0:
                        phase *= 1j
                    else:
                        phase *= -1j

                elif p == 'Z':
                    if bit_value == 1:
                        phase *= -1

            expectation += np.conj(self.state[i]) * phase * self.state[j]

        return expectation.real
    
    
    def copy(self):

        new_qc = Quantum_Circuit(self.numqubits)
        new_qc.state = self.state.copy()

        return new_qc
