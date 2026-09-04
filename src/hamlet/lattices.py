import numpy as np

def square_lattice(Nx, Ny, y_periodic = False, full_periodic = False):
    bonds = []
    for i in range(Nx * Ny):
        if (i+1) % Nx != 0: # Connects right up until boundary
            bonds.append([i, i+1])
        if i < Nx*(Ny-1): # Connects down up until boundary
            bonds.append([i, i+Nx])
    return bonds
