#!/usr/bin/env python3
"""Run the quantum-walk solver directly (no FastAPI)."""

from __future__ import annotations

import argparse

from _common import print_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--graph-type", choices=["cycle", "complete"], default="cycle")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--use-ionq", action="store_true")
    parser.add_argument("--output", choices=["summary", "json"], default="summary")
    args = parser.parse_args()

    from backend.quantum.quantum_walks import run_quantum_walk

    result = run_quantum_walk(
        n_qubits=args.n_qubits,
        n_steps=args.n_steps,
        graph_type=args.graph_type,
        use_simulator=not args.use_ionq,
        shots=args.shots,
    )
    print_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
