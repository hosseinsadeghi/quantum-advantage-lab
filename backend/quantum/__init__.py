"""Quantum Advantage Lab - Quantum Circuit Layer.

Provides quantum algorithm implementations using Qiskit with IonQ provider support.

Modules:
    provider        - Backend management (Aer simulator / IonQ hardware)
    grovers         - Grover's search algorithm
    quantum_walks   - Discrete-time quantum walks
    vqe             - Variational Quantum Eigensolver
    hamiltonian_sim - Hamiltonian simulation via Trotterization
"""

from backend.quantum.provider import get_backend
from backend.quantum.grovers import build_grover_circuit, run_grovers
from backend.quantum.quantum_walks import build_walk_circuit, run_quantum_walk
from backend.quantum.vqe import build_vqe_ansatz, build_hamiltonian, run_vqe
from backend.quantum.hamiltonian_sim import build_trotter_circuit, run_simulation

__all__ = [
    "get_backend",
    "build_grover_circuit",
    "run_grovers",
    "build_walk_circuit",
    "run_quantum_walk",
    "build_vqe_ansatz",
    "build_hamiltonian",
    "run_vqe",
    "build_trotter_circuit",
    "run_simulation",
]
