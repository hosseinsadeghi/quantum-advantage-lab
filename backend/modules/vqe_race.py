"""VQE vs Classical Optimization race module — placeholder.

Registers the race so it appears in the module listing, but both solvers
raise ``NotImplementedError`` until the underlying algorithms are filled in.
"""

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
        "Nelder-Mead optimization for molecular ground-state energies. Coming soon."
    )
    default_params: dict[str, Any] = {
        "molecule": "H2",
        "max_iterations": 50,
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_vqe(**params)

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_classical_optimization(**params)
