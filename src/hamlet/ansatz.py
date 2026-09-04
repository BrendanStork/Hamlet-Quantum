
def hardware_efficient_ansatz(qc, init_params, layers=1):
    
    num_qubits = qc.numqubits
    theta_index = 0
    
    for _ in range(layers):
        
        for q in range(num_qubits):
            qc.ry(q, init_params[theta_index])
            theta_index += 1
            
            qc.rz(q, init_params[theta_index])
            theta_index += 1
            
        for q in range(num_qubits-1):
            qc.cx(q, q+1)

    return qc
