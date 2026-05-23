import { useMemo, useState } from "react";

const MODULE_META = {
  grovers_search: {
    icon: "\u{1F50D}",
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
      use_simulator: {
        type: "toggle", label: "Use Simulator",
        help: "Run on local Qiskit Aer simulator (fast, ideal) instead of real IonQ hardware (slower, noisy but real quantum computation).",
      },
    },
  },
  quantum_walks: {
    icon: "\u{1F6B6}",
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
      use_simulator: {
        type: "toggle", label: "Use Simulator",
        help: "Run on local Qiskit Aer simulator (fast, ideal) instead of real IonQ hardware.",
      },
    },
  },
  vqe: {
    icon: "\u269B",
    label: "VQE",
    params: {
      max_iterations: {
        type: "range", min: 10, max: 200, step: 10, label: "Max Iterations",
        help: "Maximum optimization iterations. The optimizer adjusts quantum circuit parameters each iteration to minimize the molecule's energy.",
      },
      use_simulator: {
        type: "toggle", label: "Use Simulator",
        help: "Run on local Qiskit Aer simulator (fast, ideal) instead of real IonQ hardware.",
      },
    },
  },
  hamiltonian_sim: {
    icon: "\u{1F300}",
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
      use_simulator: {
        type: "toggle", label: "Use Simulator",
        help: "Run on local Qiskit Aer simulator (fast, ideal) instead of real IonQ hardware.",
      },
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
              <span className="nav-icon">{m?.icon || "\u2699"}</span>
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

  if (!meta || paramEntries.length === 0) return null;

  return (
    <div className="param-controls">
      {paramEntries.map(([key, param]) => {
        if (param.type === "toggle") {
          const val = config[key] ?? true;
          return (
            <div className="param-group" key={key}>
              <label className="param-label">
                {param.label}
                {param.help && <HelpTip text={param.help} />}
              </label>
              <div className="toggle-wrapper">
                <div
                  className={`toggle ${val ? "active" : ""}`}
                  onClick={() => onConfigChange(key, !val)}
                />
                <span className="toggle-text">{val ? "On" : "Off"}</span>
              </div>
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
