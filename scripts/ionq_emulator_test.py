#!/usr/bin/env python3
"""Submit a simple circuit to an IonQ hardware emulator.

IonQ's cloud exposes two simulator modes through the *same* ``ionq_simulator``
backend:

  * ``noise_model="ideal"`` — noiseless statevector simulator (free, instant).
  * ``noise_model="forte-1"`` (or ``aria-1`` / ``aria-2`` for the retired
    devices) — a noise-aware **emulator** that replays the chosen QPU's
    calibration (gate errors, T1/T2) over a classical simulator.  Same API
    shape as the real device, but free.  As of 2026, Forte-1 is the only
    IonQ QPU still in production; the Aria profiles remain available as
    historical emulator targets.

This script runs a Bell circuit (plus an idle period that exposes
decoherence) on both the ideal simulator and the chosen emulator, then prints
the counts side by side so you can see the noise bite.

Requires:
    * ``IONQ_API_KEY`` exported in the environment (get one at
      https://cloud.ionq.com → Settings → API Keys).
    * ``qiskit-ionq`` installed (``uv sync --extra ionq``).
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


def build_bell_with_idle(entangle_pairs: int = 10):
    """Bell pair followed by N echo pairs ``rxx(pi/2); barrier; rxx(-pi/2)``.

    Why entangling pairs and not single-qubit rotations?  On IonQ Forte-1 the
    published 1q gate fidelity is ~99.98%, so 100 ``rx`` gates contribute only
    ~2% aggregate error and the emulator's output is essentially clean.
    Noise on trapped-ion hardware is dominated by the 2q Mølmer-Sørensen
    interaction (~99.6% fidelity on Forte-1, ~0.35% per gate; ~99.3% on Aria,
    ~0.7% per gate).  A ``rxx`` echo pair is *mathematically* identity but
    consists of two full-strength MS gates on the pair; each one is subject
    to the emulator's 2q error model.
    The barrier between halves blocks the transpiler from algebraically
    merging the pair back into the identity.
    """
    import numpy as np
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2, name="bell_with_entangling_idle")
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(entangle_pairs):
        qc.rxx(np.pi / 2, 0, 1)
        qc.barrier()
        qc.rxx(-np.pi / 2, 0, 1)
        qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


def pretty_counts(counts: dict[str, int | float], shots: int) -> str:
    total = sum(counts.values()) or 1
    rows = []
    for k in sorted(counts):
        v = counts[k]
        frac = v / total
        rows.append(f"    {k} : {v:>8}  ({frac*100:5.1f}%)")
    return "\n".join(rows) if rows else "    (empty)"


def run_on(backend, circuit, *, noise_model: str, shots: int):
    """Submit to the IonQ backend and return counts.

    Ideal simulator forces shots=1 and returns probabilities; we scale to
    ``shots`` so the output is comparable to the emulator.
    """
    if noise_model == "ideal":
        job = backend.run(circuit, noise_model="ideal")
        result = job.result()
        probs = result.get_counts()
        return {k: int(round(v * shots)) for k, v in probs.items()}

    job = backend.run(circuit, noise_model=noise_model, shots=shots)
    print(f"  submitted job id={job.job_id()} (this can take ~10-60 s on the emulator)")
    result = job.result()
    return dict(result.get_counts())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--noise-model",
        choices=VALID_NOISE_MODELS,
        default="forte-1",
        help="IonQ emulator noise profile (default: forte-1, the only active QPU; "
             "aria-1/aria-2 are available as retired-device profiles).",
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--entangle-pairs", type=int, default=10,
                        help="Number of rxx(pi/2);barrier;rxx(-pi/2) echo pairs after the Bell "
                             "pair. Each pair = 2 physical MS gates on the pair; 2q gates "
                             "are what dominate ion-trap error (~0.7%% each on Aria-1). "
                             "0 = plain Bell.")
    parser.add_argument("--skip-ideal", action="store_true",
                        help="Don't also run on noise_model='ideal' for comparison.")
    args = parser.parse_args()

    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        print("IONQ_API_KEY is not set. Get one at https://cloud.ionq.com → Settings → API Keys.", file=sys.stderr)
        return 2

    try:
        from qiskit_ionq import IonQProvider  # type: ignore[import-untyped]
    except ImportError:
        print("qiskit-ionq is not installed. Run: uv sync --extra ionq", file=sys.stderr)
        return 2

    from backend.quantum.provider import transpile_for_ionq

    qc = build_bell_with_idle(args.entangle_pairs)
    qc_ionq = transpile_for_ionq(qc, optimization_level=2)

    ionq_ops = qc_ionq.count_ops()
    twoq_gates = ionq_ops.get("rxx", 0) + ionq_ops.get("cx", 0)
    expected_twoq = 1 + 2 * args.entangle_pairs  # Bell's CX + 2 per echo pair

    print(f"circuit: bell + {args.entangle_pairs} rxx echo pairs ({2*args.entangle_pairs} physical MS gates total)")
    print(f"  depth (qis)   = {qc.depth()}, ops = {dict(qc.count_ops())}")
    print(f"  depth (ionq)  = {qc_ionq.depth()}, ops = {dict(ionq_ops)}")
    print(f"  2q gates surviving transpile: {twoq_gates} / {expected_twoq}"
          + ("  (warning: echo pairs collapsed, emulator noise will be low)"
             if args.entangle_pairs > 0 and twoq_gates < expected_twoq // 2 else ""))
    # Rough expected noise floor from 2q gate error rates (ballpark).
    #   forte-1 published 2q fidelity ~99.6%  -> ~0.0035 err/gate
    #   aria-{1,2} published 2q fidelity ~99.3% -> ~0.007 err/gate
    if twoq_gates >= 2 and args.noise_model != "ideal":
        per_gate_err = 0.0035 if args.noise_model.startswith("forte") else 0.007
        expected_err = 1 - (1 - per_gate_err) ** twoq_gates
        print(f"  rough noise floor on {args.noise_model} from {twoq_gates} 2q gates: "
              f"~{expected_err*100:.1f}% total error (using ~{per_gate_err*100:.2f}%/gate)")
    print()

    provider = IonQProvider(api_key)
    backend = provider.get_backend("ionq_simulator")
    print(f"backend: {backend.name} (noise_model routed per-job)")
    print()

    if not args.skip_ideal:
        print("[1/2] noise_model=ideal (noiseless reference)")
        ideal_counts = run_on(backend, qc, noise_model="ideal", shots=args.shots)
        print(pretty_counts(ideal_counts, args.shots))
        print()

    print(f"[{'2/2' if not args.skip_ideal else '1/1'}] noise_model={args.noise_model} (emulator)")
    noisy_counts = run_on(backend, qc, noise_model=args.noise_model, shots=args.shots)
    print(pretty_counts(noisy_counts, args.shots))
    print()

    fraction_bell = (noisy_counts.get("00", 0) + noisy_counts.get("11", 0)) / max(sum(noisy_counts.values()), 1)
    print(f"P(|00> + |11>) on {args.noise_model} = {fraction_bell:.3f}  (ideal Bell: 1.000)")
    print("OK" if fraction_bell > 0.8 else "WARN: noise larger than expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
