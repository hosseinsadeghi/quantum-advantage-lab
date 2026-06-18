"""Hamiltonian Simulation vs Classical Matrix Exponentiation race module."""

from __future__ import annotations

from typing import Any

from backend.modules.base import RaceModule
from backend.quantum.hamiltonian_sim import run_simulation
from backend.quantum.provider import resolve_shots
from backend.classical.matrix_exp import run_classical_simulation

# Keys consumed directly by the race module itself; everything else is treated
# as a model coupling constant (J, h, Jx, Jy, Jz, ...) and forwarded through.
_RACE_PARAM_KEYS = frozenset({
    "n_qubits", "model", "time", "n_steps",
    "use_simulator", "use_qpu", "noise_model", "qpu_name",
    "shots", "initial_state", "seed_simulator",
})


def _model_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Return just the coupling constants to forward to the Hamiltonian builders."""
    return {k: v for k, v in params.items() if k not in _RACE_PARAM_KEYS}


class SimulationRace(RaceModule):
    module_id = "hamiltonian_sim"
    title = "Hamiltonian Simulation vs Matrix Exponentiation"
    description = (
        "Compare quantum Hamiltonian simulation (Trotter-Suzuki "
        "decomposition) against classical matrix exponentiation. "
        "Quantum simulation scales polynomially while classical "
        "methods scale exponentially in system size."
    )
    default_params: dict[str, Any] = {
        "n_qubits": 4,
        "model": "ising",
        "time": 1.0,
        "n_steps": 10,
        "interaction_pattern": "chain",
        "alpha": 3.0,
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_simulation(
            n_qubits=params.get("n_qubits", 4),
            model=params.get("model", "ising"),
            time=params.get("time", 1.0),
            n_steps=params.get("n_steps", 10),
            use_simulator=params.get("use_simulator", True),
            use_qpu=params.get("use_qpu", False),
            noise_model=params.get("noise_model", "forte-1"),
            qpu_name=params.get("qpu_name", "qpu.forte-1"),
            shots=resolve_shots(params),
            initial_state=params.get("initial_state"),
            seed_simulator=params.get("seed_simulator"),
            **_model_kwargs(params),
        )

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_classical_simulation(
            n_qubits=params.get("n_qubits", 4),
            model=params.get("model", "ising"),
            time_total=params.get("time", 1.0),
            n_steps=params.get("n_steps", 10),
            initial_state=params.get("initial_state"),
            **_model_kwargs(params),
        )
