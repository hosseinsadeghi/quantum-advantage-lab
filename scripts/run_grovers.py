#!/usr/bin/env python3
"""Run Grover's search quantum solver directly (no FastAPI)."""

from __future__ import annotations

import argparse

from _common import print_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--target-state", type=int, default=7)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--use-ionq", action="store_true")
    parser.add_argument("--output", choices=["summary", "json"], default="summary")
    args = parser.parse_args()

    from backend.quantum.grovers import run_grovers

    result = run_grovers(
        n_qubits=args.n_qubits,
        target_state=args.target_state,
        use_simulator=not args.use_ionq,
        shots=args.shots,
    )
    print_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
