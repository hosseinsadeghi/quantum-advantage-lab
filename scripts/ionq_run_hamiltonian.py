#!/usr/bin/env python3
"""Run the Trotter Hamiltonian-simulation circuit on an IonQ emulator.

This is the Hamiltonian-sim analogue of ``ionq_emulator_test.py``.  It takes our
actual workload (``build_trotter_circuit`` for a TFIM / Heisenberg chain),
transpiles it through ``transpile_for_ionq`` into the ``{rx, ry, rz, rxx}``
basis, then submits three runs so you can cleanly separate the two sources of
error:

  * **exact** — classical ``expm(-i H t)`` applied to the initial statevector.
    No shot noise, no Trotter approximation.  Ground truth.
  * **ideal circuit** — the Trotter circuit on ``ionq_simulator`` with
    ``noise_model="ideal"``.  *Only* Trotter error survives.
  * **noisy circuit** — the same circuit on ``ionq_simulator`` with
    ``noise_model="forte-1"`` (or another IonQ emulator profile — aria-1,
    aria-2 are retired-device profiles still available for reproducibility).
    Trotter error *plus* device noise.

The reported total-variation distances ``(ideal vs exact)`` and
``(noisy vs ideal)`` decompose the overall deviation into "algorithmic" vs
"device" contributions — the same split we'll visualise in the UI in Phase 2.

Requires ``IONQ_API_KEY`` and ``qiskit-ionq`` (``uv sync --extra ionq``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


VALID_NOISE_MODELS = ("ideal", "aria-1", "aria-2", "forte-1")


def _tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _normalise(counts: dict[str, int | float]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _exact_probabilities(n_qubits: int, model: str, time: float,
                         initial_state: str, **model_kwargs: float) -> dict[str, float]:
    """Classical expm(-i H t) |initial>, returned as bitstring probabilities."""
    import numpy as np
    from qiskit.quantum_info import Statevector

    from backend.quantum.hamiltonian_sim import _exact_evolution, get_model_terms

    terms = get_model_terms(n_qubits, model, **model_kwargs)
    sv0 = Statevector.from_label(initial_state)
    sv = _exact_evolution(terms, n_qubits, time, sv0)
    probs = np.abs(sv.data) ** 2
    return {format(i, f"0{n_qubits}b"): float(p) for i, p in enumerate(probs) if p > 1e-10}


def _run_on_ionq(backend, circuit, *, noise_model: str, shots: int) -> dict[str, int | float]:
    """Route a single job on the IonQ cloud simulator.

    ``noise_model="ideal"`` collapses to a statevector sim that ignores the
    caller's ``shots`` and returns probabilities; we leave those as-is for
    max resolution and normalise downstream.  For real noise models we get
    actual counts.
    """
    if noise_model == "ideal":
        job = backend.run(circuit, noise_model="ideal")
    else:
        job = backend.run(circuit, noise_model=noise_model, shots=shots)
        print(f"  submitted job id={job.job_id()} on {noise_model} (~10-60 s typical)")
    return dict(job.result().get_counts())


def _print_table(n_qubits: int, rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print distributions side-by-side, sorted by descending probability of the first column."""
    all_keys = set()
    for _, d in rows:
        all_keys.update(d)
    # Sort by first column magnitude
    primary = rows[0][1]
    keys = sorted(all_keys, key=lambda k: -primary.get(k, 0.0))

    header = " " * (n_qubits + 2) + "  ".join(f"{name:>10}" for name, _ in rows)
    print(header)
    print("-" * len(header))
    for k in keys[:16]:
        cells = "  ".join(f"{d.get(k, 0.0)*100:>9.2f}%" for _, d in rows)
        print(f"{k}  {cells}")
    if len(keys) > 16:
        print(f"  ... {len(keys) - 16} more bitstrings below 1-2% each")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-qubits", type=int, default=3,
                        help="System size (default 3 — keeps emulator run fast).")
    parser.add_argument("--model", choices=["ising", "heisenberg"], default="ising")
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=4,
                        help="Trotter steps. Higher = smaller algorithmic error, deeper circuit.")
    parser.add_argument("--initial-state", default=None,
                        help="Computational-basis bitstring, defaults to all-zeros.")
    parser.add_argument("--noise-model", choices=VALID_NOISE_MODELS, default="forte-1",
                        help="IonQ emulator profile (default forte-1 — the only active QPU). "
                             "aria-1/aria-2 available as retired-device profiles.")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--skip-exact", action="store_true",
                        help="Skip the classical expm reference.")
    parser.add_argument("--skip-ideal", action="store_true",
                        help="Skip the IonQ ideal-simulator run (still runs the emulator).")
    args = parser.parse_args()

    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        print("IONQ_API_KEY is not set. Get one at https://cloud.ionq.com -> Settings -> API Keys.",
              file=sys.stderr)
        return 2

    try:
        from qiskit_ionq import IonQProvider  # type: ignore[import-untyped]
    except ImportError:
        print("qiskit-ionq is not installed. Run: uv sync --extra ionq", file=sys.stderr)
        return 2

    from backend.quantum.hamiltonian_sim import build_trotter_circuit, get_model_terms
    from backend.quantum.provider import transpile_for_ionq, circuit_metadata

    initial_state = args.initial_state or "0" * args.n_qubits
    if len(initial_state) != args.n_qubits:
        parser.error("--initial-state length must match --n-qubits")

    # ------------------------------------------------------------------
    # Build the Trotter circuit (with initial-state prep prefixed)
    # ------------------------------------------------------------------
    from qiskit import QuantumCircuit

    terms = get_model_terms(args.n_qubits, args.model)
    trotter_body = build_trotter_circuit(terms, time=args.time, n_steps=args.n_steps)
    trotter_body.remove_final_measurements(inplace=True)

    qc = QuantumCircuit(args.n_qubits, args.n_qubits)
    for i, bit in enumerate(reversed(initial_state)):
        if bit == "1":
            qc.x(i)
    qc.compose(trotter_body, inplace=True)
    qc.measure(range(args.n_qubits), range(args.n_qubits))

    qc_ionq = transpile_for_ionq(qc, optimization_level=2)

    print(f"problem:  {args.model} chain, n_qubits={args.n_qubits}, "
          f"t={args.time}, n_steps={args.n_steps}, |psi_0>=|{initial_state}>")
    print(f"circuit:  depth(qis)={qc.depth()}  ops={dict(qc.count_ops())}")
    print(f"          depth(ionq)={qc_ionq.depth()}  ops={dict(qc_ionq.count_ops())}")
    meta = circuit_metadata(qc_ionq)
    print(f"          cx+rxx gates = {meta['gate_counts'].get('cx', 0) + meta['gate_counts'].get('rxx', 0)}, "
          f"total gates = {meta['total_gates']}")
    print()

    distributions: list[tuple[str, dict[str, float]]] = []

    if not args.skip_exact:
        exact = _exact_probabilities(
            args.n_qubits, args.model, args.time, initial_state,
        )
        distributions.append(("exact", exact))

    provider = IonQProvider(api_key)
    backend = provider.get_backend("ionq_simulator")

    if not args.skip_ideal:
        print("submitting: noise_model=ideal (Trotter-only error)")
        ideal = _normalise(_run_on_ionq(backend, qc_ionq, noise_model="ideal", shots=args.shots))
        distributions.append(("ideal", ideal))

    print(f"submitting: noise_model={args.noise_model} (Trotter + device noise)")
    noisy = _normalise(_run_on_ionq(backend, qc_ionq, noise_model=args.noise_model, shots=args.shots))
    distributions.append((args.noise_model, noisy))

    print()
    print("Probability distribution (top 16 bitstrings):")
    _print_table(args.n_qubits, distributions)
    print()

    # ------------------------------------------------------------------
    # Error decomposition
    # ------------------------------------------------------------------
    by_name = {name: d for name, d in distributions}
    if "exact" in by_name and "ideal" in by_name:
        trotter_tv = _tv_distance(by_name["ideal"], by_name["exact"])
        print(f"TV(ideal  vs exact)  = {trotter_tv:.4f}   <- Trotter algorithmic error")
    if "ideal" in by_name and args.noise_model in by_name:
        noise_tv = _tv_distance(by_name[args.noise_model], by_name["ideal"])
        print(f"TV({args.noise_model:<6} vs ideal)  = {noise_tv:.4f}   <- device (emulator) noise")
    if "exact" in by_name and args.noise_model in by_name:
        total_tv = _tv_distance(by_name[args.noise_model], by_name["exact"])
        print(f"TV({args.noise_model:<6} vs exact)  = {total_tv:.4f}   <- total observed error")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
