import { useEffect, useMemo, useState } from "react";

// VITE_BACKEND_URL is only set at build time on the hosted (Vercel) deploy,
// which points at a simulator-only backend. Locally it is unset, so QPU stays
// reachable and the `use_simulator` toggle remains interactive.
const IS_HOSTED_DEMO = !!import.meta.env.VITE_BACKEND_URL;

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
        type: "range", min: 0, max: 15, step: 1, label: "Target State",
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
      solver: SOLVER_PARAM,
    },
  },
  vqe: {
    label: "VQE",
    params: {
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
        type: "range", min: 2, max: 8, step: 1, label: "Qubits",
        help: "Number of spins in the quantum system. Classical cost grows exponentially: 2^n matrix size.",
      },
      time: {
        type: "range", min: 0.5, max: 10, step: 0.5, label: "Sim Time",
        help: "Total simulation time (arbitrary units). How long we evolve the quantum state under the Hamiltonian. Longer times show more complex dynamics.",
      },
      n_steps: {
        type: "range", min: 5, max: 50, step: 5, label: "Trotter Steps",
        help: "Number of Trotter decomposition steps. More steps = more accurate simulation but deeper quantum circuit. Trade-off between precision and noise.",
      },
      solver: SOLVER_PARAM,
    },
  },
};

/* Tooltip component */
function HelpTip({ text }) {
  const [visible, setVisible] = useState(false);

  return (
    <span
      className="help-tip-wrapper"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <span className="help-tip-icon">?</span>
      {visible && (
        <span className="help-tip-popup">{text}</span>
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

  // Live per-QPU availability (null while loading / when not applicable). Drives
  // which real-hardware options are greyed out in the solver dropdown.
  const [qpuAvail, setQpuAvail] = useState(null);
  useEffect(() => {
    if (IS_HOSTED_DEMO) return undefined;
    let cancelled = false;
    fetch("/api/backends/qpu")
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
        if (param.type === "select") {
          // The dropdown drives the backend's use_simulator/use_qpu/qpu_name
          // params directly — `solver` itself is never sent to the backend.
          const current =
            config.use_simulator === false
              ? (config.use_qpu ? (config.qpu_name || "qpu.forte-1") : "ionq_emulator")
              : "aer";
          // The hosted demo can only reach the local simulator — QPU/IonQ are
          // not wired up there, so lock the selector to Aer.
          const locked = IS_HOSTED_DEMO;
          const value = locked ? "aer" : current;
          const patchFor = (v) => {
            if (v === "ionq_emulator")
              return { use_simulator: false, use_qpu: false, noise_model: "forte-1" };
            if (v.startsWith("qpu."))
              return { use_simulator: false, use_qpu: true, qpu_name: v };
            return { use_simulator: true, use_qpu: false };
          };
          const isQpu = value.startsWith("qpu.");
          return (
            <div className="param-group" key={key}>
              <label className="param-label">
                {param.label}
                {param.help && <HelpTip text={param.help} />}
              </label>
              <select
                className="param-select"
                value={value}
                disabled={locked}
                title={locked ? "Simulator-only on the public demo" : undefined}
                onChange={(e) => onConfigChange(patchFor(e.target.value))}
              >
                {param.options.map((opt) => {
                  const avail = opt.qpu ? qpuAvail?.[opt.value] : null;
                  // Only gate once availability has loaded; never disable Aer/emulator.
                  const unavailable =
                    opt.qpu && qpuAvail != null && !avail?.available;
                  const suffix = unavailable
                    ? ` — ${avail?.reason || "unavailable"}`
                    : "";
                  return (
                    <option key={opt.value} value={opt.value} disabled={unavailable}>
                      {opt.label}
                      {suffix}
                    </option>
                  );
                })}
              </select>
              {isQpu && (
                <p className="param-warning">
                  Real IonQ hardware ({value}) — each run submits a billable job.
                </p>
              )}
            </div>
          );
        }

        const val = config[key] ?? param.min;
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
              min={param.min}
              max={param.max}
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
