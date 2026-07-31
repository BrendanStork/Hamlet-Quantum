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
        self.gates = []
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
        self.gates.append(('X', qubitIndex))
        return

        
    def y(self, qubitIndex):
        self.gate_op(GATES['Y'], qubitIndex)
        self.gates.append(('Y', qubitIndex))
        return
        
    def z(self, qubitIndex):
        self.gate_op(GATES['Z'], qubitIndex)
        self.gates.append(('Z', qubitIndex))
        return
        
    def h(self, qubitIndex):
        self.gate_op(GATES['H'], qubitIndex)
        self.gates.append(('H', qubitIndex))
        return

    def s(self, qubitIndex):
        self.gate_op(GATES['S'], qubitIndex)
        self.gates.append(('S', qubitIndex))
        return
        
    def sdag(self, qubitIndex):
        self.gate_op(GATES['Sdag'], qubitIndex)
        self.gates.append(('S\u2020', qubitIndex))
        return

    def t(self, qubitIndex):
        self.gate_op(GATES['T'], qubitIndex)
        self.gates.append(('T', qubitIndex))
        return

    def p(self, qubitIndex, theta):
        self.gate_op(GATES['P'](theta), qubitIndex)
        return

    def rx(self, qubitIndex, theta):
        self.gate_op(GATES['RX'](theta), qubitIndex)
        self.gates.append(('RX', qubitIndex, theta))
        return

    def ry(self, qubitIndex, theta):
        self.gate_op(GATES['RY'](theta), qubitIndex)
        self.gates.append(('RY', qubitIndex, theta))
        return
        
    def rz(self, qubitIndex, theta):
        self.gate_op(GATES['RZ'](theta), qubitIndex)
        self.gates.append(('RZ', qubitIndex, theta))
        return

    def cx(self, control, target):
        self.state = apply_cnot(self.state, control, target)
        self.gates.append(('CX', control, target))
        return
        

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
        
        
        
    def layers(self):

        layers = []

        for gate in self.gates:

            if gate[0] == 'CX':

                used = set(
                    range(
                        min(gate[1], gate[2]),
                        max(gate[1], gate[2]) + 1
                    )
                )

            else:

                used = {gate[1]}

            # First gate creates the first layer
            if not layers:
                layers.append([gate])
                continue

            occupied = set()

            # Only inspect the most recent layer
            for g in layers[-1]:

                if g[0] == 'CX':

                    occupied.update(
                        range(
                            min(g[1], g[2]),
                            max(g[1], g[2]) + 1
                        )
                    )

                else:

                    occupied.add(g[1])

            if occupied.isdisjoint(used):

                layers[-1].append(gate)

            else:

                layers.append([gate])

        return layers
        
    def draw(self, layers_per_block=17):

        WIDTH = 5
        circuit_line = '─' * WIDTH
        #nrows = 2 * self.numqubits-1

        layers = self.layers()

        # Draw one block at a time
        for start in range(0, len(layers), layers_per_block):
            
            lines = [
                f'q{i}: {"─"*2}'
                for i in range(self.numqubits)
            ]
            
            '''
            lines = ['' for _ in range(nrows)]

            for q in range(self.numqubits):
                lines[2 * q] = f'q{q}: {"─"*2}'
            '''
            # Only draw this block of layers
            for layer in layers[start:start + layers_per_block]:
                
                
                cells = ['─' * (WIDTH + 1) for _ in range(self.numqubits)]
                '''
                cells = [' ' * (WIDTH + 1) for _ in range(nrows)]

                for q in range(self.numqubits):
                    cells[2*q] = '─' * (WIDTH + 1)
                '''
                    
                for gate in layer:

                    name = gate[0]

                    if name != 'CX':

                        target = gate[1]

                        if name.startswith('R'):
                            label = name
                            cells[target] = f'{label}{circuit_line[:-1]}'
                        else:
                            label = name
                            cells[target] = f'{label}{circuit_line}'

                    else:

                        control = gate[1]
                        target = gate[2]

                        low = min(control, target)
                        high = max(control, target)

                        cells[control] = f'●{circuit_line}'
                        cells[target] = f'X{circuit_line}'

                        for q in range(low + 1, high):
                            cells[q] = f'│{circuit_line}'

                for q in range(self.numqubits):
                    lines[q] += cells[q]

            for line in lines:
                print(line)

            # Blank line between blocks
            if start + layers_per_block < len(layers):
                print()
    

    
