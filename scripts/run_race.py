#!/usr/bin/env python3
"""Run a full quantum-vs-classical race via the RaceModule class."""

from __future__ import annotations

import argparse
import asyncio
import json

from _common import _json_default

MODULE_CHOICES = ["grovers_search", "quantum_walks", "vqe", "hamiltonian_sim"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=MODULE_CHOICES, required=True)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a param, e.g. --param n_qubits=3. Repeatable.",
    )
    parser.add_argument("--use-ionq", action="store_true",
                        help="Route quantum run to the IonQ cloud emulator (ionq_simulator).")
    parser.add_argument("--use-qpu", action="store_true",
                        help="Route to the real IonQ QPU. Billable. Implies --use-ionq.")
    parser.add_argument("--noise-model", default="forte-1",
                        help="Emulator profile when --use-ionq is set (default: forte-1; "
                             "also: ideal, aria-1, aria-2).")
    parser.add_argument("--add-shots", type=int, metavar="N",
                        help="Run an extra N shots and append as a new cache record "
                             "(bypasses lookup).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the result cache for this run (still writes on "
                             "cacheable runs).")
    parser.add_argument("--output", choices=["summary", "json"], default="summary")
    args = parser.parse_args()

    from backend.modules import MODULES

    params: dict = {}
    for kv in args.param:
        if "=" not in kv:
            parser.error(f"--param expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        params[k] = _parse_value(v)

    if args.use_ionq or args.use_qpu:
        params["use_simulator"] = False
    if args.use_qpu:
        params["use_qpu"] = True
    params["noise_model"] = args.noise_model
    if args.no_cache:
        params["cache_bypass"] = True
    if args.add_shots is not None:
        params["add_shots"] = True
        params["shots"] = args.add_shots

    mod = MODULES[args.module]()
    result = asyncio.run(mod.run(params))

    if args.output == "json":
        payload = {
            "module": args.module,
            "params": params,
            "quantum": {
                "steps": result.quantum_steps,
                "result": result.quantum_result,
                "time": result.quantum_time,
            },
            "classical": {
                "steps": result.classical_steps,
                "result": result.classical_result,
                "time": result.classical_time,
            },
        }
        print(json.dumps(payload, default=_json_default, indent=2))
        return 0

    print(f"module: {args.module}")
    print(f"params: {params or '(defaults)'}")
    print(f"quantum: steps={len(result.quantum_steps)} time={result.quantum_time:.4f}s")
    _print_final(result.quantum_result, prefix="  q")
    print(f"classical: steps={len(result.classical_steps)} time={result.classical_time:.4f}s")
    _print_final(result.classical_result, prefix="  c")
    return 0


def _parse_value(v: str):
    vl = v.lower()
    if vl in {"true", "false"}:
        return vl == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _print_final(res: dict, prefix: str) -> None:
    for k, v in (res or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            print(f"{prefix}.{k}: {v}")


if __name__ == "__main__":
    raise SystemExit(main())
