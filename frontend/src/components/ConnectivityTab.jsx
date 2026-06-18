import { useEffect, useMemo, useState } from "react";

const DEFAULT_PARAMS = {
  n_qubits: 4,
  model: "ising",
  time: 1,
  n_steps: 10,
  interaction_pattern: "chain",
  alpha: 3,
};

function extractAnalysisParams(config) {
  return {
    n_qubits: Number(config?.n_qubits ?? DEFAULT_PARAMS.n_qubits),
    model: config?.model || DEFAULT_PARAMS.model,
    time: Number(config?.time ?? DEFAULT_PARAMS.time),
    n_steps: Number(config?.n_steps ?? DEFAULT_PARAMS.n_steps),
    interaction_pattern: config?.interaction_pattern || DEFAULT_PARAMS.interaction_pattern,
    alpha: Number(config?.alpha ?? DEFAULT_PARAMS.alpha),
  };
}

function formatNumber(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return "--";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(digits);
}

function MetricCard({ label, value, suffix = "", tone = "neutral" }) {
  return (
    <div className={`connectivity-metric-card tone-${tone}`}>
      <div className="connectivity-metric-label">{label}</div>
      <div className="connectivity-metric-value">
        {formatNumber(value)}
        {suffix}
      </div>
    </div>
  );
}

function InteractionGraph({ graph }) {
  const width = 250;
  const height = 210;
  const cx = width / 2;
  const cy = height / 2;
  const radius = 74;
  const n = graph?.n_qubits || 0;
  const edges = graph?.edges || [];
  const maxWeight = Math.max(...edges.map((edge) => edge.weight), 1);

  const nodes = useMemo(() => {
    return Array.from({ length: n }, (_, index) => {
      const angle = (-Math.PI / 2) + (index * 2 * Math.PI) / Math.max(n, 1);
      return {
        index,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      };
    });
  }, [n, cx, cy, radius]);

  if (!graph) {
    return <div className="placeholder">Waiting for interaction graph...</div>;
  }

  return (
    <div className="connectivity-graph-shell">
      <svg viewBox={`0 0 ${width} ${height}`} className="connectivity-graph">
        {edges.map((edge) => {
          const source = nodes[edge.source];
          const target = nodes[edge.target];
          if (!source || !target) return null;
          const opacity = 0.2 + 0.8 * (edge.weight / maxWeight);
          return (
            <line
              key={`${edge.source}-${edge.target}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="rgba(14,116,144,0.9)"
              strokeWidth={1 + 3 * (edge.weight / maxWeight)}
              strokeOpacity={opacity}
            />
          );
        })}
        {nodes.map((node) => (
          <g key={node.index}>
            <circle
              cx={node.x}
              cy={node.y}
              r="14"
              fill="rgba(67, 97, 238, 0.18)"
              stroke="rgba(0, 212, 255, 0.9)"
              strokeWidth="1.5"
            />
            <text
              x={node.x}
              y={node.y + 4}
              textAnchor="middle"
              fill="#e5eef4"
              fontSize="10"
              fontFamily="'JetBrains Mono', monospace"
            >
              q{node.index}
            </text>
          </g>
        ))}
      </svg>
      <div className="connectivity-caption">
        {graph.pattern === "power_law"
          ? `Power-law couplings with alpha=${formatNumber(graph.alpha, 2)}`
          : graph.pattern === "all_to_all"
            ? "Every qubit pair is coupled directly"
            : "Nearest-neighbor chain interactions"}
      </div>
    </div>
  );
}

function ComparisonBars({ ionq, heavyHex }) {
  const groups = [
    { label: "Depth", ionq: ionq?.depth, heavy: heavyHex?.depth },
    { label: "2Q Gates", ionq: ionq?.two_qubit_gates, heavy: heavyHex?.two_qubit_gates },
    { label: "SWAPs", ionq: ionq?.swap_count, heavy: heavyHex?.swap_count },
  ];
  const maxValue = Math.max(
    1,
    ...groups.flatMap((group) => [group.ionq || 0, group.heavy || 0]),
  );

  return (
    <div className="connectivity-bars">
      {groups.map((group) => (
        <div className="connectivity-bar-group" key={group.label}>
          <div className="connectivity-bar-group-title">{group.label}</div>
          <div className="connectivity-bar-row">
            <span className="connectivity-bar-label">IonQ</span>
            <div className="connectivity-bar-track">
              <div
                className="connectivity-bar-fill ionq"
                style={{ width: `${((group.ionq || 0) / maxValue) * 100}%` }}
              />
            </div>
            <span className="connectivity-bar-value">{formatNumber(group.ionq, 0)}</span>
          </div>
          <div className="connectivity-bar-row">
            <span className="connectivity-bar-label">Heavy-hex</span>
            <div className="connectivity-bar-track">
              <div
                className="connectivity-bar-fill heavy"
                style={{ width: `${((group.heavy || 0) / maxValue) * 100}%` }}
              />
            </div>
            <span className="connectivity-bar-value">{formatNumber(group.heavy, 0)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ConnectivityTab({ config, apiBase, layout = "panel" }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const analysisParams = useMemo(() => extractAnalysisParams(config), [config]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch(`${apiBase}/api/analysis/hamiltonian/connectivity`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: analysisParams }),
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || payload.message || "Connectivity analysis failed.");
        }
        return payload;
      })
      .then((payload) => {
        setAnalysis(payload);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setError(err.message || "Connectivity analysis failed.");
        setLoading(false);
      });

    return () => controller.abort();
  }, [apiBase, analysisParams]);

  const logical = analysis?.logical;
  const ionq = analysis?.ionq;
  const heavyHex = analysis?.heavy_hex;
  const metrics = analysis?.metrics;
  const graph = analysis?.interaction_graph;
  const isPage = layout === "page";

  return (
    <div className={`connectivity-page ${isPage ? "page-layout" : "panel-layout"}`}>
      {isPage && (
        <div className="connectivity-page-hero">
          <div>
            <div className="connectivity-page-kicker">Hamiltonian Analysis</div>
            <h2 className="connectivity-page-title">Connectivity Advantage</h2>
            <p className="connectivity-page-copy">
              This page follows the current Hamiltonian Simulation parameters from the sidebar,
              then transpiles the same circuit to IonQ&apos;s all-to-all topology and a constrained
              heavy-hex baseline to compare routing overhead directly.
            </p>
          </div>
        </div>
      )}

      <div className={`connectivity-page-grid ${isPage ? "wide" : ""}`}>
        <div className="connectivity-column connectivity-column-main">
          <div className="anno-section">
            <h4 className="anno-section-title">Current Simulation Setup</h4>
            <div className="connectivity-config-chips">
              <div className="connectivity-config-chip">
                <span>Qubits</span>
                <code>{formatNumber(analysisParams.n_qubits, 0)}</code>
              </div>
              <div className="connectivity-config-chip">
                <span>Model</span>
                <code>{analysisParams.model}</code>
              </div>
              <div className="connectivity-config-chip">
                <span>Time</span>
                <code>{formatNumber(analysisParams.time, 1)}</code>
              </div>
              <div className="connectivity-config-chip">
                <span>Trotter</span>
                <code>{formatNumber(analysisParams.n_steps, 0)}</code>
              </div>
              <div className="connectivity-config-chip">
                <span>Pattern</span>
                <code>{analysisParams.interaction_pattern}</code>
              </div>
              {analysisParams.interaction_pattern === "power_law" && (
                <div className="connectivity-config-chip">
                  <span>Alpha</span>
                  <code>{formatNumber(analysisParams.alpha, 1)}</code>
                </div>
              )}
            </div>
            <p className="connectivity-inline-note">
              Change these from the normal Hamiltonian Simulation controls in the sidebar.
            </p>
          </div>

          <div className="anno-section">
            <h4 className="anno-section-title">IonQ vs Heavy-hex</h4>
            {loading && <div className="placeholder">Transpiling both targets...</div>}
            {error && <div className="placeholder">{error}</div>}
            {!loading && !error && analysis && (
              <>
                <div className="connectivity-metrics-grid">
                  <MetricCard
                    label="Routing Depth Reduction"
                    value={metrics?.routing_depth_reduction_pct}
                    suffix="%"
                    tone="good"
                  />
                  <MetricCard
                    label="SWAP Tax Avoided"
                    value={metrics?.swap_tax_avoided}
                    tone="good"
                  />
                  <MetricCard
                    label="2Q Overhead Reduction"
                    value={metrics?.two_qubit_overhead_reduction_pct}
                    suffix="%"
                    tone="good"
                  />
                </div>

                <div className="connectivity-target-strip">
                  <div className="connectivity-target-chip ionq">
                    IonQ all-to-all
                    <code>d={formatNumber(ionq?.depth, 0)}</code>
                  </div>
                  <div className="connectivity-target-chip heavy">
                    Heavy-hex
                    <code>
                      d={formatNumber(heavyHex?.depth, 0)} / distance {formatNumber(heavyHex?.topology?.distance, 0)}
                    </code>
                  </div>
                </div>

                <ComparisonBars ionq={ionq} heavyHex={heavyHex} />
              </>
            )}
          </div>
        </div>

        <div className="connectivity-column">
          <div className="anno-section">
            <h4 className="anno-section-title">Logical Reference</h4>
            {logical ? (
              <div className="connectivity-reference-grid">
                <div className="connectivity-reference-card">
                  <span className="connectivity-reference-label">Logical depth</span>
                  <code>{formatNumber(logical.depth, 0)}</code>
                </div>
                <div className="connectivity-reference-card">
                  <span className="connectivity-reference-label">Logical 2Q depth</span>
                  <code>{formatNumber(logical.two_qubit_depth, 0)}</code>
                </div>
                <div className="connectivity-reference-card">
                  <span className="connectivity-reference-label">Logical 2Q gates</span>
                  <code>{formatNumber(logical.two_qubit_gates, 0)}</code>
                </div>
              </div>
            ) : (
              <div className="placeholder">Logical circuit stats will appear after analysis.</div>
            )}
          </div>

          <div className="anno-section">
            <h4 className="anno-section-title">Interaction Graph</h4>
            <InteractionGraph graph={graph} />
          </div>
        </div>
      </div>
    </div>
  );
}
