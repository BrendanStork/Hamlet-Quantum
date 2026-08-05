import numpy as np

from many_body_qsim.lattices import square_lattice
from many_body_qsim.hamiltonians import transverse_ising_hamiltonian
from many_body_qsim.circuits import Quantum_Circuit
from many_body_qsim.observables import magnetization, observable_vs_time


def test_tfim_exact_vs_trotter_workflow():

    # Build lattice
    Nx = 3
    Ny = 2

    bonds = square_lattice(
        Nx=Nx,
        Ny=Ny
    )

    # Build Hamiltonian
    H = transverse_ising_hamiltonian(
        bonds,
        J=1.0,
        h=1.0,
        axis='X'
    )

    # Initialize state
    N = Nx * Ny

    qc = Quantum_Circuit(N)

    qc.x(0)

    # Observable
    mZ = magnetization(axis='Z')

    # Evolution
    time = 5
    timesteps = 20

    t, mag_exact = observable_vs_time(
        qc,
        H,
        time=time,
        timesteps=timesteps,
        method='exact',
        observable=mZ
    )

    _, mag_trotter = observable_vs_time(
        qc,
        H,
        time=time,
        timesteps=timesteps,
        method='trotter_fixed_steps',
        trotter_steps=10,
        observable=mZ
    )


    # Assertions

    # Output lengths match
    assert len(t) == timesteps

    assert len(mag_exact) == len(t)

    assert len(mag_trotter) == len(t)

    # Results are finite
    assert np.all(np.isfinite(mag_exact))
    assert np.all(np.isfinite(mag_trotter))
