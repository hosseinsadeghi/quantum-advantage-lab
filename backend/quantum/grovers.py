"""Grover's search — placeholder.

The quantum side of the Grover-vs-linear-search race. Will implement a
parameterized Grover circuit (phase-flip oracle + Grover diffusion) with
intermediate statevector snapshots. Not yet implemented; this module exposes
the public API shape so the race registry can load.
"""

from __future__ import annotations

from typing import Any


def build_grover_circuit(*args: Any, **kwargs: Any):  # pragma: no cover - stub
    raise NotImplementedError("Grover's circuit builder is not yet implemented.")


def run_grovers(*args: Any, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - stub
    raise NotImplementedError("Grover's search solver is not yet implemented.")
