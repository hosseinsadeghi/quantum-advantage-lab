"""Quantum Advantage Lab — Qollab cell: Hamiltonian Simulation convergence demo.

Single-file version of the Hamiltonian Simulation race for the Qollab
project page. For each value in N_STEPS_SWEEP, builds a first-order Trotter
circuit approximating evolution of |0...0> under a 1D transverse-field Ising
Hamiltonian, runs it on the pre-created ``backend`` (default: Qiskit
BasicSimulator; switch to IonQ via the QPU dropdown), and compares the
sampled distribution to the exact distribution from scipy.linalg.expm.

The platform calls ``main(shots, excludeLowProbabilityValues,
lowProbabilityThreshold)`` with values from the form below the editor,
so this file does not invoke ``main`` itself.
"""

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

# ---- Problem parameters (edit these to explore) -----------------------------
N_QUBITS = 4
EVOLUTION_TIME = 0.5
J_COUPLING = 1.0
TRANSVERSE_FIELD = 1.0
N_STEPS_SWEEP = [1, 2, 4, 8, 16]

# QAVE (Quantum Algorithm Visualization Engine, q-inho/qave) animation will
# fire only when the "QAVE" checkbox in Qollab is ticked (which installs the
# `qave` package). The animation is rendered for a single representative
# Trotter circuit at this many steps — kept small so frame count stays sane.
QAVE_ANIMATION_STEPS = 2


def build_ising_hamiltonian(n_qubits: int, J: float, h: float) -> SparsePauliOp:
    """1D open-chain transverse-field Ising:  H = -J Σ Z_i Z_{i+1} - h Σ X_i."""
    terms: list[tuple[str, float]] = []
    for i in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[i] = "Z"
        label[i + 1] = "Z"
        terms.append(("".join(reversed(label)), -J))
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[i] = "X"
        terms.append(("".join(reversed(label)), -h))
    return SparsePauliOp.from_list(terms)


def exact_distribution(H: SparsePauliOp, t: float, n_qubits: int) -> dict[str, float]:
    """Exact |<x|exp(-iHt)|0...0>|^2 for every computational-basis state x."""
    U = expm(-1j * H.to_matrix() * t)
    psi0 = np.zeros(2 ** n_qubits, dtype=complex)
    psi0[0] = 1.0
    psi = U @ psi0
    probs = np.abs(psi) ** 2
    return {format(i, f"0{n_qubits}b"): float(p) for i, p in enumerate(probs)}


def trotter_circuit(H: SparsePauliOp, t: float, n_steps: int, n_qubits: int) -> QuantumCircuit:
    """First-order (Lie-Trotter) circuit approximating exp(-iHt)."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.append(
        PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=n_steps)),
        range(n_qubits),
    )
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def counts_to_probs(counts: dict[str, int], n_qubits: int) -> dict[str, float]:
    total = sum(counts.values()) or 1
    probs = {format(i, f"0{n_qubits}b"): 0.0 for i in range(2 ** n_qubits)}
    for bitstring, c in counts.items():
        key = bitstring.replace(" ", "").zfill(n_qubits)[-n_qubits:]
        probs[key] = probs.get(key, 0.0) + c / total
    return probs


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def render_with_qave(circuit: QuantumCircuit, n_steps: int, shots: int) -> None:
    """Emit a QAVE animation of `circuit` to ./qave_artifacts.

    QAVE (https://github.com/q-inho/qave) is enabled by ticking the "QAVE"
    checkbox above the editor on Qollab, which installs the `qave` package
    along with the Processing + ffmpeg renderer. If the package is absent
    (checkbox off) this function silently no-ops. If the package is present
    but the renderer is missing, we fall back to a deterministic trace.json.
    """
    try:
        from qave import (  # type: ignore[import-not-found]
            ArtifactOptions,
            RenderOptions,
            SimulationOptions,
            generate_animation_from_qiskit,
            generate_trace_from_qiskit,
        )
    except ImportError:
        return  # QAVE feature not enabled in this Qollab session.

    from pathlib import Path

    # Flatten PauliEvolutionGate into elementary gates so QAVE renders the
    # internal Trotter structure (ZZ blocks + X-rotation layers) rather than
    # a single opaque "evolution" box.
    flattened = circuit.decompose(reps=3)

    sim_opts = SimulationOptions(
        algorithm_id="custom",
        mode="preview",
        seed=24,
        shot_count=max(shots, 1),
    )
    artifact_opts = ArtifactOptions(out_dir=Path("qave_artifacts"))
    render_opts = RenderOptions(
        width=640,
        height=360,
        fps=15,
        keep_frames=False,
        emit_mp4=False,
        emit_gif=True,
    )

    print(f"\n[QAVE] Rendering Trotter circuit (n_steps={n_steps}) ...")
    try:
        result = generate_animation_from_qiskit(
            flattened, options=sim_opts, render=render_opts, artifacts=artifact_opts,
        )
    except Exception as exc:  # noqa: BLE001 - renderer dep errors vary
        print(f"[QAVE] Animation renderer failed ({exc.__class__.__name__}: {exc}).")
        print("[QAVE] Falling back to deterministic trace.json only.")
        try:
            trace = generate_trace_from_qiskit(
                flattened, options=sim_opts, artifacts=artifact_opts,
            )
            print(f"[QAVE] Trace written to {trace.paths.trace_json}")
        except Exception as exc2:  # noqa: BLE001
            print(f"[QAVE] Trace generation also failed: {exc2}")
        return

    if result.gif_path is not None:
        print(f"[QAVE] GIF: {result.gif_path}")
    print(f"[QAVE] Trace: {result.paths.trace_json}")


def main(
    shots: int = 1024,
    excludeLowProbabilityValues: bool = True,
    lowProbabilityThreshold: float = 0.05,
):
    H = build_ising_hamiltonian(N_QUBITS, J_COUPLING, TRANSVERSE_FIELD)
    exact = exact_distribution(H, EVOLUTION_TIME, N_QUBITS)

    backend_name = getattr(backend, "name", str(backend))  # type: ignore[name-defined]  # noqa: F821
    print(f"Transverse-field Ising chain  n={N_QUBITS}, J={J_COUPLING}, h={TRANSVERSE_FIELD}")
    print(f"Evolution time t = {EVOLUTION_TIME}")
    print(f"Backend: {backend_name}")
    print(f"Shots per Trotter circuit: {shots}\n")

    header = f"{'n_steps':>8} | {'depth':>6} | {'2q gates':>9} | {'TV(quantum, exact)':>22}"
    print(header)
    print("-" * len(header))

    last_probs: dict[str, float] = {}
    for n_steps in N_STEPS_SWEEP:
        qc = trotter_circuit(H, EVOLUTION_TIME, n_steps, N_QUBITS)
        tqc = transpile(qc, backend, optimization_level=1)  # type: ignore[name-defined]  # noqa: F821
        result = backend.run(tqc, shots=shots).result()  # type: ignore[name-defined]  # noqa: F821
        probs = counts_to_probs(result.get_counts(), N_QUBITS)

        twoq = sum(1 for inst in tqc.data if len(inst.qubits) >= 2)
        tvd = tv_distance(probs, exact)
        print(f"{n_steps:>8} | {tqc.depth():>6} | {twoq:>9} | {tvd:>22.4f}")
        last_probs = probs

    print(
        "\nDistribution at largest n_steps "
        f"(n_steps={N_STEPS_SWEEP[-1]}) vs exact reference:"
    )
    for bitstring in sorted(exact, key=lambda b: -exact[b]):
        p_exact = exact[bitstring]
        p_quant = last_probs.get(bitstring, 0.0)
        if excludeLowProbabilityValues and max(p_exact, p_quant) < lowProbabilityThreshold:
            continue
        print(f"  |{bitstring}>   exact={p_exact:.4f}   trotter={p_quant:.4f}")

    # Optional QAVE animation — enable the "QAVE" checkbox in Qollab to install
    # the `qave` package; this no-ops otherwise. Render at a small step count so
    # the frame budget stays modest.
    H_for_anim = build_ising_hamiltonian(N_QUBITS, J_COUPLING, TRANSVERSE_FIELD)
    anim_circuit = trotter_circuit(
        H_for_anim, EVOLUTION_TIME, QAVE_ANIMATION_STEPS, N_QUBITS
    )
    render_with_qave(anim_circuit, n_steps=QAVE_ANIMATION_STEPS, shots=shots)

    # Optional matplotlib figure — enable the "Visualization" checkbox to see it.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = sorted(exact)
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, [exact[k] for k in labels], width, label="Exact (expm)")
    ax.bar(
        x + width / 2,
        [last_probs.get(k, 0.0) for k in labels],
        width,
        label=f"Trotter (n_steps={N_STEPS_SWEEP[-1]})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Probability")
    ax.set_title("Trotter vs exact under transverse-field Ising")
    ax.legend()
    fig.tight_layout()
    fig.savefig("trotter_vs_exact.png", dpi=120)
    print("\nSaved convergence chart to trotter_vs_exact.png")
