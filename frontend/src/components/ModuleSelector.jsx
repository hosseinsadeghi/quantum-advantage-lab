import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/$/, "");
const QPU_SUBMISSION_DISABLED =
  /^(1|true|yes|on)$/i.test(import.meta.env.VITE_DISABLE_QPU_SUBMISSION || "");

// Execution targets exposed by the backend provider (see backend/quantum/provider.py).
// QPU option values are the literal IonQ device names so the patch can forward
// them straight through as `qpu_name`:
//   aer                     -> use_simulator=true                       (local, free, ideal)
//   ionq_emulator           -> use_simulator=false, use_qpu=false        (cloud sim, free, noise-modeled)
//   qpu.forte-1             -> use_simulator=false, use_qpu=true, qpu_name (real HW, billable)
//   qpu.forte-enterprise-1  -> use_simulator=false, use_qpu=true, qpu_name (real HW, billable)
const SOLVER_PARAM = {
  type: "select",
  label: "Solver",
  options: [
    { value: "aer", label: "Local simulator (Aer)" },
    { value: "ionq_emulator", label: "IonQ emulator (forte-1)" },
    { value: "qpu.forte-1", label: "IonQ QPU — forte-1 (real HW)", qpu: true },
    { value: "qpu.forte-enterprise-1", label: "IonQ QPU — forte-enterprise-1 (real HW)", qpu: true },
  ],
  help: "Where the quantum circuit runs. Local Aer is fast and ideal. IonQ emulator is a free cloud simulator with forte-1 noise. The two QPUs are real trapped-ion hardware — billable and slower; unavailable devices are greyed out.",
};

const MODULE_META = {
  grovers_search: {
    label: "Grover's Search",
    params: {
      n_qubits: {
        type: "range", min: 2, max: 12, step: 1, label: "Qubits",
        help: "Number of qubits. The search space has 2^n items — e.g. 4 qubits = 16 items to search through.",
      },
      target_state: {
        type: "range",
        min: 0,
        max: (config) => Math.max(0, 2 ** (config.n_qubits || 4) - 1),
        step: 1,
        label: "Target State",
        help: "The item we're searching for, represented as an integer index. Grover's algorithm will try to find this state.",
      },
      solver: SOLVER_PARAM,
    },
  },
  quantum_walks: {
    label: "Quantum Walk",
    params: {
      n_qubits: {
        type: "range", min: 2, max: 6, step: 1, label: "Qubits",
        help: "Number of qubits for position encoding. The graph has 2^n nodes — e.g. 3 qubits = 8-node graph.",
      },
      n_steps: {
        type: "range", min: 5, max: 50, step: 5, label: "Walk Steps",
        help: "How many steps the walker takes on the graph. More steps show the spreading behavior more clearly.",
      },
      graph_type: {
        type: "select",
        label: "Graph Type",
        options: [
          { value: "cycle", label: "Cycle" },
          { value: "complete", label: "Complete" },
        ],
        help: "Cycle uses a single coin qubit and nearest-neighbor motion. Complete uses a larger coin register and allows hopping across the full graph.",
      },
      n_trials: {
        type: "range", min: 100, max: 5000, step: 100, label: "Classical Trials",
        help: "Number of Monte Carlo trials for the classical random walk reference.",
      },
      solver: SOLVER_PARAM,
    },
  },
  vqe: {
    label: "VQE",
    params: {
      molecule: {
        type: "select",
        label: "Molecule",
        options: [
          { value: "H2", label: "H2" },
          { value: "LiH", label: "LiH" },
        ],
        help: "Problem Hamiltonian for the VQE benchmark.",
      },
      n_layers: {
        type: "range", min: 1, max: 5, step: 1, label: "Ansatz Layers",
        help: "Depth of the hardware-efficient ansatz. More layers increase expressivity but also circuit depth.",
      },
      max_iterations: {
        type: "range", min: 10, max: 200, step: 10, label: "Max Iterations",
        help: "Maximum optimization iterations. The optimizer adjusts quantum circuit parameters each iteration to minimize the molecule's energy.",
      },
      solver: SOLVER_PARAM,
    },
  },
  hamiltonian_sim: {
    label: "Hamiltonian Sim",
    params: {
      n_qubits: {
        type: "range", min: 2, max: 12, step: 1, label: "Qubits",
        help: "Number of spins in the quantum system. Classical cost grows exponentially: 2^n matrix size.",
      },
      time: {
        type: "range", min: 0.5, max: 10, step: 0.5, label: "Sim Time",
        help: "Total simulation time (arbitrary units). How long we evolve the quantum state under the Hamiltonian. Longer times show more complex dynamics.",
      },
      model: {
        type: "select",
        label: "Model",
        options: [
          { value: "ising", label: "Ising" },
          { value: "heisenberg", label: "Heisenberg" },
        ],
        help: "Hamiltonian family to simulate. Ising mixes ZZ couplings with a transverse X field. Heisenberg evolves XX, YY, and ZZ couplings together.",
      },
      n_steps: {
        type: "range", min: 5, max: 50, step: 5, label: "Trotter Steps",
        help: "Number of Trotter decomposition steps. More steps = more accurate simulation but deeper quantum circuit. Trade-off between precision and noise.",
      },
      interaction_pattern: {
        type: "select",
        label: "Coupling Pattern",
        options: [
          { value: "chain", label: "Chain" },
          { value: "power_law", label: "Power-law" },
          { value: "all_to_all", label: "All-to-all" },
        ],
        help: "Which qubit pairs interact in the simulated Hamiltonian. Dense patterns make routing pressure on constrained hardware more visible.",
      },
      alpha: {
        type: "range", min: 0.5, max: 4, step: 0.1, label: "Power-law Alpha",
        help: "Decay exponent for the power-law interaction pattern. Only used when Coupling Pattern is set to Power-law.",
        visible: (config) => config.interaction_pattern === "power_law",
      },
      solver: SOLVER_PARAM,
    },
  },
};

/* Tooltip component */
function HelpTip({ text }) {
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState(null);
  const anchorRef = useRef(null);

  useEffect(() => {
    if (!visible || !anchorRef.current) return undefined;

    const updatePosition = () => {
      if (!anchorRef.current) return;
      const rect = anchorRef.current.getBoundingClientRect();
      const popupWidth = 240;
      const popupHeight = 150;
      const gutter = 12;

      let left = rect.right + 10;
      if (left + popupWidth > window.innerWidth - gutter) {
        left = Math.max(gutter, rect.left - popupWidth - 10);
      }

      let top = rect.top + rect.height / 2;
      const minTop = gutter + popupHeight / 2;
      const maxTop = window.innerHeight - gutter - popupHeight / 2;
      top = Math.min(Math.max(top, minTop), maxTop);

      setPosition({ left, top });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [visible]);

  return (
    <span
      ref={anchorRef}
      className="help-tip-wrapper"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <span className="help-tip-icon">?</span>
      {visible && position && (
        <span
          className="help-tip-popup"
          style={{ left: `${position.left}px`, top: `${position.top}px` }}
        >
          {text}
        </span>
      )}
    </span>
  );
}

/* Sidebar navigation list */
function ModuleSelector({ modules, selected, onSelect }) {
  return (
    <ul className="nav-list">
      {modules.map((mod) => {
        const m = MODULE_META[mod.id];
        const isActive = selected?.id === mod.id;
        return (
          <li key={mod.id}>
            <button
              className={`nav-item ${isActive ? "active" : ""}`}
              onClick={() => onSelect(mod)}
            >
              <span className="nav-label">{m?.label || mod.title}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/* Parameter controls (rendered separately in sidebar) */
function Controls({ moduleId, config, onConfigChange }) {
  const meta = moduleId ? MODULE_META[moduleId] : null;

  const paramEntries = useMemo(() => {
    if (!meta) return [];
    return Object.entries(meta.params);
  }, [meta]);

  useEffect(() => {
    if (!meta) return;
    const patches = {};

    for (const [key, param] of Object.entries(meta.params)) {
      if (param.visible && !param.visible(config)) continue;
      if (param.type !== "range") continue;

      const min = typeof param.min === "function" ? param.min(config) : param.min;
      const max = typeof param.max === "function" ? param.max(config) : param.max;
      const current = config[key];
      if (current == null) continue;

      const clamped = Math.min(Math.max(current, min), max);
      if (clamped !== current) {
        patches[key] = clamped;
      }
    }

    if (Object.keys(patches).length > 0) {
      onConfigChange(patches);
    }
  }, [config, meta, onConfigChange]);

  // Live per-QPU availability (null while loading / when not applicable). Shown
  // as an informational note next to each device; it does NOT block selection —
  // IonQ enforces authorization at submission time, not this dropdown.
  const [qpuAvail, setQpuAvail] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/backends/qpu`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled && d) setQpuAvail(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  if (!meta || paramEntries.length === 0) return null;

  return (
    <div className="param-controls">
      {paramEntries.map(([key, param]) => {
        if (param.visible && !param.visible(config)) {
          return null;
        }
        if (param.type === "select") {
          const isSolver = key === "solver";
          const current =
            config.use_simulator === false
              ? (config.use_qpu ? (config.qpu_name || "qpu.forte-1") : "ionq_emulator")
              : "aer";
          const value = isSolver
            ? current
            : (config[key] ?? param.options[0]?.value ?? "");
          const patchFor = (v) => {
            if (v === "ionq_emulator") {
              return { use_simulator: false, use_qpu: false, noise_model: "forte-1" };
            }
            if (v.startsWith("qpu.")) {
              return { use_simulator: false, use_qpu: true, qpu_name: v };
            }
            return { use_simulator: true, use_qpu: false };
          };
          const isQpu = isSolver && value.startsWith("qpu.");
          return (
            <div className="param-group" key={key}>
              <label className="param-label">
                {param.label}
                {param.help && <HelpTip text={param.help} />}
              </label>
              <select
                className="param-select"
                value={value}
                onChange={(e) => (
                  isSolver
                    ? onConfigChange(patchFor(e.target.value))
                    : onConfigChange(key, e.target.value)
                )}
              >
                {param.options.map((opt) => {
                  if (!isSolver) {
                    return (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    );
                  }
                  const avail = opt.qpu ? qpuAvail?.[opt.value] : null;
                  // Disable only on an explicit device-down status. Missing
                  // access or missing credentials must not block selection,
                  // because a cached hardware result can still be replayed.
                  const deviceDown = opt.qpu
                    && qpuAvail != null
                    && avail
                    && ["offline", "unavailable", "retired", "maintenance", "calibrating"].includes(
                      String(avail.status || "").toLowerCase()
                    );
                  let note = "";
                  if (opt.qpu && qpuAvail != null && avail) {
                    if (avail.status !== "available") note = ` — ${avail.status}`;
                    else if (!avail.has_access)
                      note = " — active key lacks access (will error unless cached)";
                  }
                  return (
                    <option key={opt.value} value={opt.value} disabled={deviceDown}>
                      {opt.label}
                      {note}
                    </option>
                  );
                })}
              </select>
              {isQpu && (
                <p className="param-warning">
                  {QPU_SUBMISSION_DISABLED
                    ? `Real IonQ hardware (${value}) selected for cache lookup only. If no cached result exists, the run will fail instead of submitting a new hardware job.`
                    : `Real IonQ hardware (${value}) — each run submits a billable job.`}
                </p>
              )}
            </div>
          );
        }

        const min = typeof param.min === "function" ? param.min(config) : param.min;
        const max = typeof param.max === "function" ? param.max(config) : param.max;
        const fallback = min ?? 0;
        const val = config[key] ?? fallback;
        return (
          <div className="param-group" key={key}>
            <label className="param-label">
              <span className="param-label-text">
                {param.label}
                {param.help && <HelpTip text={param.help} />}
              </span>
              <span className="param-value">{val}</span>
            </label>
            <input
              type="range"
              min={min}
              max={max}
              step={param.step}
              value={val}
              onChange={(e) =>
                onConfigChange(
                  key,
                  param.step % 1 === 0 ? parseInt(e.target.value) : parseFloat(e.target.value)
                )
              }
            />
          </div>
        );
      })}
    </div>
  );
}

ModuleSelector.Controls = Controls;

export default ModuleSelector;
