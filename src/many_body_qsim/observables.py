import numpy as np
from .evolution import trotter_evolve, trotter_step, exact_evolve
from .circuits import Quantum_Circuit



def magnetization(axis='Z'):

    def _obs(qc):
        N = int(np.log2(len(qc.state)))  # number of qubits
        total = 0

        for i in range(N):
            pauli_term = ['I'] * N
            pauli_term[i] = axis
            joined_pauli_term = ''.join(pauli_term)

            total += qc.expectation_value(joined_pauli_term)

        return total / N

    return _obs
 
def total_expectation_value(qc, total_pauli_string):
    length_pauli_term = len(next(iter(total_pauli_string)))
    exp_val = 0
    for pauli_term, coeff in H.items():
        exp_val += coeff * qc.expectation_value(pauli_term)

        
    return exp_val

def expectation_value(operator_string):
    def _obs(psi):
        return psi.expectation_value(operator_string)
    return _obs


def two_site_correlation(i, j, axis='Z'):

    def _obs(psi):
        N = psi.numqubits
        op = ['I'] * N
        if i != j:
            op[i] = axis
            op[j] = axis
        joined_op = ''.join(op)
        return psi.expectation_value(joined_op)

    return _obs
    
def correlation_map(axis='Z'):

    def _obs(psi):
        N = psi.numqubits
        corr = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                op = ['I'] * N
                if i != j:
                    op[i] = axis
                    op[j] = axis
                corr[i, j] = psi.expectation_value(''.join(op))

        return corr

    return _obs
    
def observable_vs_time(qc0, basis, *, time, timesteps, method, observable, dt = None, sample_every = 1, trotter_steps = None):
    N = timesteps
    dt = time/timesteps
    t = np.linspace(0, time, N)
    
    vals = np.zeros(N)

    if method == 'exact':
        for i in range(N):
            qc = qc0.copy()
            psi = exact_evolve(qc, basis, time = t[i])
            vals[i] = observable(psi)

    elif method == 'trotter':
        qc = qc0.copy()
        #print(N)
        sample_length = int(N/sample_every)
        #print(sample_length)
        vals = []
        for i in range(0, N):
            qc = trotter_step(qc, basis, dt = dt)
            if i % sample_every == 0:
                #print(i % sample_every)
                vals.append(observable(qc))
    elif method == 'trotter_fixed_steps':
        for i in range(timesteps):
            qc = qc0.copy()
            psi = trotter_evolve(
                qc,
                basis,
                time=t[i],
                trotter_steps=trotter_steps
            )
            vals[i] = observable(psi)
    else:
        raise ValueError("Unknown method")

    return t, vals
    
def observable_vs_trottersteps(qc0, basis, *, time, observable, trottersteps):
    steps = np.arange(1, trottersteps+1)
    vals = np.zeros(trottersteps)

    for i in range(trottersteps):
        qc = qc0.copy()
        psi = trotter_evolve(qc, basis, time = time, trotter_steps=steps[i])
        vals[i] = observable(psi)

    return steps, vals
    
    
def fidelity(state1, state2):
    return np.abs(np.vdot(state1, state2))**2
