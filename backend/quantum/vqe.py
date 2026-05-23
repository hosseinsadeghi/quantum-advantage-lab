"""Variational Quantum Eigensolver — placeholder.

The quantum side of the VQE-vs-classical-optimization race. Will implement a
hardware-efficient ansatz (Ry/Rz rotation layers + all-to-all RXX entanglers,
native to IonQ's Molmer-Sorensen gate) optimized via COBYLA against H2 / LiH
Bravyi-Kitaev Hamiltonians. Not yet implemented.
"""

from __future__ import annotations

from typing import Any


def build_vqe_ansatz(*args: Any, **kwargs: Any):  # pragma: no cover - stub
    raise NotImplementedError("VQE ansatz builder is not yet implemented.")


def build_hamiltonian(*args: Any, **kwargs: Any):  # pragma: no cover - stub
    raise NotImplementedError("VQE Hamiltonian builder is not yet implemented.")


def run_vqe(*args: Any, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - stub
    raise NotImplementedError("VQE solver is not yet implemented.")
