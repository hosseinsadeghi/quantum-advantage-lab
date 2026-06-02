"""VQE vs Classical Gradient Optimization race module."""

from __future__ import annotations

from typing import Any

from backend.modules.base import RaceModule
from backend.quantum.vqe import run_vqe
from backend.classical.gradient_opt import run_classical_optimization


class VQERace(RaceModule):
    module_id = "vqe"
    title = "VQE vs Classical Optimization"
    description = (
        "Compare the Variational Quantum Eigensolver against classical "
        "gradient-based optimization for finding molecular ground-state "
        "energies. VQE leverages quantum superposition to explore the "
        "energy landscape."
    )
    default_params: dict[str, Any] = {
        "molecule": "H2",
        "n_layers": 2,
        "max_iterations": 100,
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_vqe(
            molecule=params.get("molecule", "H2"),
            n_layers=params.get("n_layers", 2),
            max_iterations=params.get("max_iterations", 100),
            use_simulator=params.get("use_simulator", True),
            use_qpu=params.get("use_qpu", False),
            noise_model=params.get("noise_model", "forte-1"),
            qpu_name=params.get("qpu_name", "qpu.forte-1"),
        )

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_classical_optimization(
            molecule=params.get("molecule", "H2"),
            max_iterations=params.get("max_iterations", 100),
        )
