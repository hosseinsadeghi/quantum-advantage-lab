"""Discrete-time quantum walks — placeholder.

The quantum side of the quantum-walk-vs-random-walk race. Will implement a
coined discrete-time quantum walk (Hadamard / Grover coin + shift operators)
on cycle and complete graphs, tracking per-step position distributions via
partial trace. Not yet implemented.
"""

from __future__ import annotations

from typing import Any


def build_walk_circuit(*args: Any, **kwargs: Any):  # pragma: no cover - stub
    raise NotImplementedError("Quantum walk circuit builder is not yet implemented.")


def run_quantum_walk(*args: Any, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - stub
    raise NotImplementedError("Quantum walk solver is not yet implemented.")
