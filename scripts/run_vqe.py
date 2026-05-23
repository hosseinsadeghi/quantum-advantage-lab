#!/usr/bin/env python3
"""Run the VQE quantum solver directly (no FastAPI)."""

from __future__ import annotations

import argparse

from _common import print_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=["H2", "LiH"], default="H2")
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--use-ionq", action="store_true")
    parser.add_argument("--output", choices=["summary", "json"], default="summary")
    args = parser.parse_args()

    from backend.quantum.vqe import run_vqe

    result = run_vqe(
        molecule=args.molecule,
        n_layers=args.n_layers,
        max_iterations=args.max_iterations,
        use_simulator=not args.use_ionq,
    )
    print_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
