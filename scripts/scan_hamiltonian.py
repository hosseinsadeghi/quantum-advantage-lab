#!/usr/bin/env python3
"""Scan Hamiltonian-simulation parameters through the race/cache layer.

By default this uses the local Aer simulator. For real QPU scans, pass
``--use-qpu``. QPU mode is cache-only unless ``--allow-submit`` is also set:
cached points replay from .cache/runs, while cache misses fail instead of
submitting fresh hardware jobs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time as time_module
from itertools import product
from pathlib import Path
from typing import Any

from _common import _json_default


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-qubits", default="2,3",
                        help="Comma-separated qubit counts, e.g. 2,3,4.")
    parser.add_argument("--models", default="ising",
                        help="Comma-separated models: ising,heisenberg.")
    parser.add_argument("--times", default="0.5,1.0",
                        help="Comma-separated evolution times.")
    parser.add_argument("--steps", default="2,4,8",
                        help="Comma-separated Trotter step counts.")
    parser.add_argument("--patterns", default="chain",
                        help="Comma-separated interaction patterns: chain,all_to_all,power_law.")
    parser.add_argument("--alphas", default="3.0",
                        help="Comma-separated alpha values used for power_law.")
    parser.add_argument("--initial-state", default=None,
                        help="Computational-basis bitstring. Default: all zeros.")
    parser.add_argument("--shots", type=int, default=None,
                        help="Shot count. QPU defaults to QAL_QPU_SHOTS or 256 if omitted.")
    parser.add_argument("--use-ionq", action="store_true",
                        help="Use IonQ cloud emulator instead of local Aer.")
    parser.add_argument("--noise-model", default="forte-1",
                        help="IonQ emulator noise model, default forte-1.")
    parser.add_argument("--use-qpu", action="store_true",
                        help="Use real IonQ QPU. Billable when --allow-submit is set.")
    parser.add_argument("--qpu-name", default="qpu.forte-1")
    parser.add_argument("--allow-submit", action="store_true",
                        help="Allow fresh QPU submissions on cache misses.")
    parser.add_argument("--add-shots", action="store_true",
                        help="Append a fresh run to an existing cache key. This submits again.")
    parser.add_argument("--output", choices=["summary", "json", "csv"], default="summary")
    parser.add_argument("--out-file", default=None,
                        help="Optional path for json/csv output. Summary always prints to stdout.")
    parser.add_argument("--submission-log", default=".cache/scan_hamiltonian_submissions.jsonl",
                        help="JSONL log for non-blocking QPU submissions.")
    args = parser.parse_args()

    if args.use_qpu and not args.allow_submit:
        os.environ["DISABLE_QPU_SUBMISSION"] = "true"

    combos = list(product(
        _ints(args.n_qubits),
        _strings(args.models),
        _floats(args.times),
        _ints(args.steps),
        _strings(args.patterns),
        _floats(args.alphas),
    ))
    param_sets = [_params_from_combo(args, combo) for combo in combos]

    validation_errors = _validate_all(param_sets)
    if validation_errors:
        rows = [_error_row(params, error) for params, error in validation_errors]
        if args.output == "json":
            _write_or_print(args.out_file, json.dumps(rows, default=_json_default, indent=2))
        elif args.output == "csv":
            _write_or_print(args.out_file, _rows_to_csv(rows))
        else:
            print(
                f"Parameter validation failed for {len(rows)} of "
                f"{len(param_sets)} combination(s); no simulations were run.",
                file=sys.stderr,
            )
            for idx, row in enumerate(rows, start=1):
                _print_row(idx, len(param_sets), row)
        return 2

    rows: list[dict[str, Any]] = []
    failures = 0

    for idx, params in enumerate(param_sets, start=1):
        if args.use_qpu:
            row = _handle_qpu_scan_point(params, allow_submit=args.allow_submit)
            if row["status"] == "submitted":
                _append_submission_log(args.submission_log, row)
        else:
            row = asyncio.run(_run_one(params))
        rows.append(row)
        if row["status"] == "error":
            failures += 1

        if args.output == "summary":
            _print_row(idx, len(combos), row)

    if args.output == "summary" and args.use_qpu:
        counts = {status: sum(1 for row in rows if row["status"] == status)
                  for status in ("cached", "submitted", "error")}
        print(
            "QPU scan summary: "
            f"cached={counts['cached']} submitted={counts['submitted']} "
            f"errors={counts['error']} submission_log={args.submission_log}"
        )

    if args.output == "json":
        _write_or_print(args.out_file, json.dumps(rows, default=_json_default, indent=2))
    elif args.output == "csv":
        text = _rows_to_csv(rows)
        _write_or_print(args.out_file, text)

    return 1 if failures else 0


def _params_from_combo(args: argparse.Namespace, combo: tuple[int, str, float, int, str, float]) -> dict[str, Any]:
    n_qubits, model, time, n_steps, pattern, alpha = combo
    params: dict[str, Any] = {
        "n_qubits": n_qubits,
        "model": model,
        "time": time,
        "n_steps": n_steps,
        "interaction_pattern": pattern,
        "alpha": alpha,
        "use_simulator": not (args.use_ionq or args.use_qpu),
        "use_qpu": args.use_qpu,
        "noise_model": args.noise_model,
        "qpu_name": args.qpu_name,
    }
    if args.initial_state is not None:
        params["initial_state"] = args.initial_state
    if args.shots is not None:
        params["shots"] = args.shots
    if args.add_shots:
        params["add_shots"] = True
    return params


def _validate_all(param_sets: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    errors: list[tuple[dict[str, Any], str]] = []
    for params in param_sets:
        try:
            _validate_params(params)
        except Exception as exc:  # noqa: BLE001
            errors.append((params, f"{type(exc).__name__}: {exc}"))
    return errors


def _validate_params(params: dict[str, Any]) -> None:
    """Validate one scan point without running a solver or touching backends."""
    from backend.quantum.hamiltonian_sim import get_model_terms

    n_qubits = params["n_qubits"]
    time = params["time"]
    n_steps = params["n_steps"]
    alpha = params["alpha"]
    shots = params.get("shots")
    initial_state = params.get("initial_state")

    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if shots is not None and shots < 1:
        raise ValueError("shots must be >= 1")
    if initial_state is not None:
        if len(initial_state) != n_qubits:
            raise ValueError("initial_state length must match n_qubits")
        invalid = set(initial_state) - {"0", "1"}
        if invalid:
            raise ValueError("initial_state must contain only 0 and 1")

    get_model_terms(
        n_qubits,
        params["model"],
        interaction_pattern=params["interaction_pattern"],
        alpha=alpha,
    )


async def _run_one(params: dict[str, Any]) -> dict[str, Any]:
    from backend.modules.simulation_race import SimulationRace

    mod = SimulationRace()
    try:
        result = await mod.run(params)
    except Exception as exc:  # noqa: BLE001
        return {
            **_param_summary(params),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    q_final = result.quantum_result or {}
    q_meta = result.quantum_metadata or {}
    execution = q_final.get("execution") or {}

    return {
        **_param_summary(params),
        "status": "ok",
        "cache_hit": bool(q_meta.get("cache_hit")),
        "cache_records": q_meta.get("cache_records"),
        "cache_shots": q_meta.get("cache_shots"),
        "backend_requested": execution.get("requested"),
        "backend_actual": execution.get("actual"),
        "backend_fell_back": execution.get("fell_back"),
        "quantum_time_s": result.quantum_time,
        "classical_time_s": result.classical_time,
        "quantum_steps": len(result.quantum_steps),
        "classical_steps": len(result.classical_steps),
        "final_fidelity": q_final.get("final_fidelity"),
        "trotter_error_reference": q_final.get("trotter_error_scaling_reference"),
        "counts": q_final.get("measured_counts"),
    }


def _handle_qpu_scan_point(params: dict[str, Any], *, allow_submit: bool) -> dict[str, Any]:
    """Replay cached QPU points; submit cache misses without waiting for results."""
    from backend.modules.simulation_race import SimulationRace

    mod = SimulationRace()
    cache_status = mod.cache_status(params)
    if cache_status["has_cached_result"] and not params.get("add_shots"):
        return {
            **_param_summary(params),
            "status": "cached",
            "cache_hit": True,
            "cache_records": cache_status["records"],
            "cache_shots": cache_status["shots"],
            "cache_key": cache_status["matched_key"] or cache_status["key"],
            "backend_requested": params["qpu_name"],
            "backend_actual": params["qpu_name"],
            "backend_fell_back": False,
        }

    if not allow_submit:
        return {
            **_param_summary(params),
            "status": "error",
            "cache_hit": False,
            "cache_key": cache_status["key"],
            "error": "cache miss and --allow-submit was not set",
        }

    try:
        submitted = _submit_qpu_without_waiting(params, cache_key=cache_status["key"])
    except Exception as exc:  # noqa: BLE001
        return {
            **_param_summary(params),
            "status": "error",
            "cache_hit": False,
            "cache_key": cache_status["key"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return submitted


def _submit_qpu_without_waiting(params: dict[str, Any], *, cache_key: str) -> dict[str, Any]:
    from qiskit import transpile

    from backend.modules.simulation_race import _model_kwargs
    from backend.quantum.hamiltonian_sim import build_problem_circuit
    from backend.quantum.provider import (
        get_backend,
        resolve_shots,
        transpile_for_ionq,
    )

    full_circuit, _ = build_problem_circuit(
        n_qubits=params["n_qubits"],
        model=params["model"],
        time=params["time"],
        n_steps=params["n_steps"],
        initial_state=params.get("initial_state"),
        **_model_kwargs(params),
    )
    full_circuit = transpile_for_ionq(full_circuit)
    backend, run_kwargs, execution = get_backend(
        use_simulator=False,
        use_qpu=True,
        noise_model=params.get("noise_model", "forte-1"),
        qpu_name=params.get("qpu_name", "qpu.forte-1"),
    )
    transpiled = transpile(full_circuit, backend=backend)
    shots = resolve_shots(params)
    job = backend.run(transpiled, shots=shots, **run_kwargs)

    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        job_id = job_id()
    job_id = str(job_id or getattr(job, "_job_id", "") or "")

    return {
        **_param_summary({**params, "shots": shots}),
        "status": "submitted",
        "cache_hit": False,
        "cache_key": cache_key,
        "job_id": job_id,
        "submitted_at_unix": time_module.time(),
        "backend_requested": execution.get("requested"),
        "backend_actual": execution.get("actual"),
        "backend_fell_back": execution.get("fell_back"),
        "circuit_depth": transpiled.depth(),
        "circuit_gate_counts": dict(transpiled.count_ops()),
    }


def _append_submission_log(path: str, row: dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "submitted",
        "timestamp_unix": row.get("submitted_at_unix"),
        "job_id": row.get("job_id"),
        "cache_key": row.get("cache_key"),
        "backend": row.get("backend_actual"),
        "params": _param_summary(row),
        "circuit": {
            "depth": row.get("circuit_depth"),
            "gate_counts": row.get("circuit_gate_counts"),
        },
    }
    with log_path.open("a") as f:
        f.write(json.dumps(record, default=_json_default, separators=(",", ":")) + "\n")


def _error_row(params: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **_param_summary(params),
        "status": "error",
        "error": error,
    }


def _param_summary(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_qubits": params["n_qubits"],
        "model": params["model"],
        "time": params["time"],
        "n_steps": params["n_steps"],
        "interaction_pattern": params["interaction_pattern"],
        "alpha": params["alpha"],
        "initial_state": params.get("initial_state"),
        "shots": params.get("shots"),
        "use_qpu": params["use_qpu"],
        "qpu_name": params["qpu_name"],
        "noise_model": params["noise_model"],
    }


def _print_row(idx: int, total: int, row: dict[str, Any]) -> None:
    label = (
        f"[{idx}/{total}] n={row['n_qubits']} model={row['model']} "
        f"t={row['time']} steps={row['n_steps']} pattern={row['interaction_pattern']}"
    )
    if row["status"] != "ok":
        if row["status"] == "cached":
            print(
                f"{label} CACHED records={row.get('cache_records')} "
                f"shots={row.get('cache_shots')} key={str(row.get('cache_key', ''))[:12]}"
            )
            return
        if row["status"] == "submitted":
            print(
                f"{label} SUBMITTED job={row.get('job_id')} "
                f"backend={row.get('backend_actual')} key={str(row.get('cache_key', ''))[:12]}"
            )
            return
        print(f"{label} ERROR {row['error']}")
        return
    cache = " cache-hit" if row["cache_hit"] else ""
    fidelity = row["final_fidelity"]
    fidelity_s = f"{fidelity:.6f}" if isinstance(fidelity, (int, float)) else "-"
    print(
        f"{label} fidelity={fidelity_s} "
        f"q_time={row['quantum_time_s']:.3f}s backend={row['backend_actual']}{cache}"
    )


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    import io

    fieldnames = [
        "status", "error", "job_id", "cache_key", "n_qubits", "model", "time", "n_steps",
        "interaction_pattern", "alpha", "initial_state", "shots", "use_qpu",
        "qpu_name", "noise_model", "cache_hit", "cache_records", "cache_shots",
        "backend_requested", "backend_actual", "backend_fell_back",
        "quantum_time_s", "classical_time_s", "quantum_steps", "classical_steps",
        "final_fidelity", "trotter_error_reference", "submitted_at_unix",
        "circuit_depth", "circuit_gate_counts",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _write_or_print(path: str | None, text: str) -> None:
    if path:
        Path(path).write_text(text)
    else:
        print(text)


def _strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _ints(value: str) -> list[int]:
    return [int(x) for x in _strings(value)]


def _floats(value: str) -> list[float]:
    return [float(x) for x in _strings(value)]


if __name__ == "__main__":
    raise SystemExit(main())
