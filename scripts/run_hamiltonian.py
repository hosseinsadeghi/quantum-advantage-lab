#!/usr/bin/env python3
"""Run the Hamiltonian-simulation quantum solver directly (no FastAPI)."""

from __future__ import annotations

import argparse

from _common import print_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--model", choices=["ising", "heisenberg"], default="ising")
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--initial-state", default=None)
    parser.add_argument("--use-ionq", action="store_true", help="Route via IonQ simulator")
    parser.add_argument("--output", choices=["summary", "json"], default="summary")
    args = parser.parse_args()

    from backend.quantum.hamiltonian_sim import run_simulation

    result = run_simulation(
        n_qubits=args.n_qubits,
        model=args.model,
        time=args.time,
        n_steps=args.n_steps,
        use_simulator=not args.use_ionq,
        shots=args.shots,
        initial_state=args.initial_state,
    )
    print_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
