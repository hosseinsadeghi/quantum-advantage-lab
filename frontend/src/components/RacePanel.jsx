import SearchHeatmap from "../visualizations/SearchHeatmap.jsx";
import GraphTraversal from "../visualizations/GraphTraversal.jsx";
import EnergyLandscape from "../visualizations/EnergyLandscape.jsx";
import MatrixDecay from "../visualizations/MatrixDecay.jsx";

const VIZ_MAP = {
  grovers_search: SearchHeatmap,
  quantum_walks: GraphTraversal,
  vqe: EnergyLandscape,
  hamiltonian_sim: MatrixDecay,
};

function formatNumber(n) {
  if (n == null) return "--";
  if (typeof n === "number") {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toLocaleString();
  }
  return String(n);
}

export default function RacePanel({
  type,
  moduleId,
  steps,
  result,
  raceState,
  viewIndex,
  sharedScale,
}) {
  const isQuantum = type === "quantum";
  const label = isQuantum ? "Quantum" : "Classical";

  let status = "setup";
  if (raceState === "running") status = "running";
  if (raceState === "complete") status = "complete";

  const VizComponent = moduleId ? VIZ_MAP[moduleId] : null;

  // Extract metadata from the latest step or result
  const latestStep = steps.length > 0 ? steps[Math.min(viewIndex, steps.length - 1)] : null;

  let statSteps = steps.length > 0 ? Math.min(viewIndex + 1, steps.length) : 0;
  let statDepth = "--";
  let statExtra = "--";

  // result comes from the complete message: {final_result, metadata, steps_count}
  const md = result?.metadata;
  if (md) {
    if (isQuantum) {
      statDepth = formatNumber(md.circuit_depth || md.depth);
      statExtra = formatNumber(md.gate_count || md.gates);
    } else {
      statDepth = formatNumber(md.total_flops_estimate || md.flop_count || md.flops);
      statExtra = md.complexity || "--";
    }
  }

  return (
    <div
      className={`race-panel ${type}`}
    >
      <div className="panel-header">
        <div className="solver-type">
          <div className={`solver-dot`} style={{
            background: isQuantum ? "var(--quantum-color)" : "var(--classical-color)",
            boxShadow: isQuantum
              ? "0 0 8px rgba(14,116,144,0.5)"
              : "0 0 8px rgba(234,88,12,0.5)",
          }} />
          <span className="solver-name">{label} Solver</span>
        </div>
        <span className={`status-badge ${status}`}>
          {status === "running" && <span className="pulse" />}
          {status}
        </span>
      </div>

      <div className="progress-bar">
        <div
          className={`progress-fill ${type}`}
          style={{
            width:
              raceState === "complete"
                ? "100%"
                : raceState === "running"
                  ? `${Math.min(((viewIndex + 1) / Math.max(steps.length, 1)) * 100, 95)}%`
                  : "0%",
          }}
        />
      </div>

      <div className="panel-stats">
        <div className="panel-stat">
          <div className="stat-label">Steps</div>
          <div className="stat-value">{statSteps}</div>
        </div>
        <div className="panel-stat">
          <div className="stat-label">{isQuantum ? "Circuit Depth" : "FLOPs"}</div>
          <div className="stat-value">{statDepth}</div>
        </div>
        <div className="panel-stat">
          <div className="stat-label">{isQuantum ? "Gate Count" : "Complexity"}</div>
          <div className="stat-value">{statExtra}</div>
        </div>
      </div>

      <div className="panel-visualization">
        {VizComponent && steps.length > 0 ? (
          <VizComponent
            steps={steps}
            viewIndex={viewIndex}
            result={result}
            type={type}
            sharedScale={sharedScale}
          />
        ) : (
          <div className="placeholder">
            {raceState === "running" ? (
              <div>
                <div className="loading-shimmer" style={{ width: 200, height: 12, marginBottom: 8 }} />
                <div className="loading-shimmer" style={{ width: 150, height: 12 }} />
              </div>
            ) : (
              "Waiting for race to start..."
            )}
          </div>
        )}
      </div>

    </div>
  );
}
