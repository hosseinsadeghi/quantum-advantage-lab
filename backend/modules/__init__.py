"""Race modules for Quantum Advantage Lab.

Each module pairs a quantum algorithm with its classical counterpart
and exposes a streaming race interface.
"""

from backend.modules.search_race import SearchRace
from backend.modules.walk_race import WalkRace
from backend.modules.vqe_race import VQERace
from backend.modules.simulation_race import SimulationRace

MODULES: dict[str, type] = {
    "grovers_search": SearchRace,
    "quantum_walks": WalkRace,
    "vqe": VQERace,
    "hamiltonian_sim": SimulationRace,
}

__all__ = ["MODULES", "SearchRace", "WalkRace", "VQERace", "SimulationRace"]
