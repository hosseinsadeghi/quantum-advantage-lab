"""Quantum Advantage Lab - Classical Solver Layer.

Provides classical algorithm counterparts for the quantum race modules,
enabling side-by-side performance comparison.

Modules:
    linear_search   - Brute-force linear search (vs Grover's)
    random_walk     - Classical random walk Monte Carlo (vs quantum walks)
    gradient_opt    - Classical gradient-based molecular optimization (vs VQE)
    matrix_exp      - Classical matrix exponentiation (vs Hamiltonian simulation)
"""

from backend.classical.linear_search import run_linear_search
from backend.classical.random_walk import run_classical_walk
from backend.classical.gradient_opt import run_classical_optimization
from backend.classical.matrix_exp import run_classical_simulation

__all__ = [
    "run_linear_search",
    "run_classical_walk",
    "run_classical_optimization",
    "run_classical_simulation",
]
