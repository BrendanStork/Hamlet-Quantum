import numpy as np


GATES = {
    'I' : np.array([[1,0], [0,1]]),
    'X' : np.array([[0, 1], [1, 0]]),
    'Y' : np.array([[0, -1j], [1j, 0]]),
    'Z' : np.array([[1, 0], [0,-1]]),
    'H' : 1/np.sqrt(2) * np.array([[1, 1], [1,-1]]),
    'S' : np.array([[1,0], [0, 1j]]),
    'Sdag' : np.array([[1,0], [0, -1j]]),
    'T' : np.array([[1,0], [0, np.exp(1j*np.pi/4)]]),
    'P' : lambda theta : np.array([[1, 0],[0, np.e**(1j*theta)]]),
    'RX': lambda theta : np.array([[np.cos(theta/2),-1j*np.sin(theta/2)], [-1j*np.sin(theta/2),np.cos(theta/2)]]),
    'RY': lambda theta : np.array([[np.cos(theta/2),-np.sin(theta/2)], [np.sin(theta/2),np.cos(theta/2)]]),
    'RZ': lambda theta : np.array([[np.exp(-1j*theta/2),0], [0,np.exp(1j*theta/2)]])
}

def apply_cnot(state, control, target):
    n = int(len(state).bit_length() - 1)  # number of qubits
    new_state = state.copy()

    if control == target:
        raise ValueError("Control and target must differ")

    # You switch the indices from big endian to little (n - 1 = max index)
    control_bit = n - 1 - control
    target_bit = n - 1 - target

    for i in range(len(state)):

        # Step 1: shift control bit to LSB
        shifted = i >> control_bit

        # Step 2: isolate that bit
        control_value = shifted & 1

        if control_value == 1:

            # Step 3: flip target bit
            flipped_i = i ^ (1 << target_bit)

            # Step 4: swap once
            if i < flipped_i:
                new_state[i], new_state[flipped_i] = (
                    state[flipped_i],
                    state[i],
                )

    return new_state

def apply_hadamard_all(qc):
    for i in range(qc.numqubits):
        qc.h(i)
    return qc
