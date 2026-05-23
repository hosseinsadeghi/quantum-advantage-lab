import { useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";

const MODULE_CONTENT = {
  grovers_search: {
    problem: "Given an unsorted collection of N items, find a specific target item. Classically you must check items one by one — on average N/2 checks, worst case N. Is there a faster way if we can query the collection in superposition?",
    getInsight: (step, viewIndex) => {
      if (!step) return "Grover's algorithm uses amplitude amplification to boost the probability of measuring the correct answer.";
      const iter = step.iteration ?? viewIndex;
      if (iter === 0)
        return "Every state starts with equal amplitude 1/√N. The search space is a flat landscape — no item is more likely than any other.";
      if (iter === 1)
        return `Iteration 1: The oracle flips the phase of the target state (marks it), then the diffusion operator reflects all amplitudes about the mean. The target's amplitude grows while others shrink slightly.`;
      if (iter <= 3)
        return `Iteration ${iter}: Each oracle+diffusion cycle pumps more amplitude into the target. The target probability is now visibly higher than the rest. Grover's needs only ~√N total iterations.`;
      return `Iteration ${iter}: Amplitude amplification is near its peak. After π/4·√N iterations the target is measured with high probability. Beyond that, the amplitude oscillates back down — more is not always better.`;
    },
    getDetails: (step) => ({
      "Oracle": "Phase-flip oracle marks the target: |x⟩ → (-1)^f(x)|x⟩, where f(x)=1 only for the target.",
      "Diffusion": "H⊗n · (2|0⟩⟨0| - I) · H⊗n — reflects amplitudes about their mean.",
      "Gate decomposition": "Oracle uses multi-controlled Z → CNOT + T gates. Diffusion uses Hadamard + X + multi-controlled Z.",
      "Optimal iterations": "⌊π/4 · √N⌋ iterations, then measure.",
      "Complexity": "O(√N) queries vs classical O(N). Provably optimal for unstructured search.",
      "Error rate": step?.error_rate ? `${(step.error_rate * 100).toFixed(2)}%` : "Simulator (ideal)",
    }),
  },

  quantum_walks: {
    problem: "A walker starts at one node of a graph and moves randomly along edges. How quickly does it explore the graph? Classically, a random walker spreads diffusively — standard deviation grows as √t. Quantum mechanics allows interference between paths, fundamentally changing how fast the walker spreads and finds targets.",
    getInsight: (step, viewIndex) => {
      if (!step) return "A quantum walk replaces random coin flips with quantum superposition, allowing the walker to take all paths simultaneously and interfere constructively toward its goal.";
      const s = step.step ?? viewIndex;
      if (s === 0)
        return "The walker is localized at node 0 — 100% probability at the starting position. The quantum coin (Hadamard gate) will create a superposition of directions before each step.";
      if (s <= 3)
        return `Step ${s}: The walker's probability is spreading outward. Notice how the quantum distribution is already broader than the classical one. Quantum interference amplifies paths that move outward and suppresses those that double back.`;
      if (s <= 8)
        return `Step ${s}: The quantum walker shows ballistic spreading — the wave front moves linearly with time (standard deviation ~ t). The classical walker is still spreading diffusively (~ √t). This gap is the quantum advantage.`;
      return `Step ${s}: The quantum walk has explored much more of the graph in the same number of steps. This faster spreading underlies quantum speedups for spatial search, element distinctness, and graph connectivity problems.`;
    },
    getDetails: (step) => ({
      "Coin operator": "Hadamard gate on the coin qubit — creates equal superposition of 'left' and 'right' before each step.",
      "Shift operator": "Conditional increment/decrement on position register, controlled by coin state.",
      "Spreading rate": "Quantum: σ(t) ∝ t (ballistic). Classical: σ(t) ∝ √t (diffusive).",
      "Hitting time": "Time to reach a target node: quantum achieves quadratic speedup on many graph types.",
      "Applications": "Spatial search O(√N), element distinctness O(N^{2/3}), graph connectivity testing.",
      "IonQ advantage": "All-to-all connectivity avoids SWAP overhead for the conditional shift operator.",
    }),
  },

  vqe: {
    problem: "Find the ground-state energy of a molecule — the lowest energy its electrons can have. This determines chemical stability, reaction rates, and material properties. Exact classical computation scales exponentially with electron count, making large molecules intractable.",
    getInsight: (step, viewIndex) => {
      if (!step) return "VQE uses a quantum circuit (ansatz) to prepare trial wavefunctions, measures their energy, then a classical optimizer adjusts the circuit parameters to minimize energy.";
      const iter = step.iteration ?? viewIndex;
      const energy = step.energy;
      if (iter === 0)
        return `Starting with random circuit parameters. Initial energy estimate: ${energy != null ? energy.toFixed(4) + " Ha" : "pending"}. The dashed green line shows the exact ground-state energy we're trying to reach.`;
      if (iter <= 5)
        return `Iteration ${iter}: Energy = ${energy != null ? energy.toFixed(4) : "?"} Ha. The classical optimizer (COBYLA) is adjusting the rotation angles in the quantum circuit, descending the energy landscape.`;
      if (iter <= 20)
        return `Iteration ${iter}: Energy = ${energy != null ? energy.toFixed(5) : "?"} Ha. Converging toward the ground state. The green band shows chemical accuracy (±1.6 mHa) — reaching it means the result is useful for real chemistry.`;
      return `Iteration ${iter}: Energy = ${energy != null ? energy.toFixed(6) : "?"} Ha. The quantum circuit is encoding electron correlations that would require exponentially many classical parameters. This is where quantum advantage emerges for larger molecules.`;
    },
    getDetails: (step) => ({
      "Ansatz": "Hardware-efficient ansatz: Ry/Rz rotation layers + all-to-all RXX entangling layers (native to IonQ's MS gate).",
      "Hamiltonian": "Molecular Hamiltonian in second quantization, mapped to qubits via Jordan-Wigner transform.",
      "Optimizer": "COBYLA — gradient-free, robust to shot noise from finite measurements.",
      "Chemical accuracy": "±1.6 mHa (1 kcal/mol) — the threshold for chemically meaningful predictions.",
      "Current params": step?.params?.length > 0 ? `[${step.params.map((p) => p.toFixed(3)).join(", ")}]` : "--",
      "Scaling advantage": "Classical exact diagonalization: O(2^n). VQE: polynomial circuit depth + classical optimization.",
    }),
  },

  hamiltonian_sim: {
    problem: "Simulate how a quantum system evolves over time under its governing Hamiltonian. This is central to understanding magnetism, superconductivity, and quantum chemistry. Classically, storing the quantum state requires 2^n complex numbers — exponentially expensive. A quantum computer can simulate this natively.",
    getInsight: (step, viewIndex) => {
      if (!step) return "The time evolution operator e^{-iHt} is approximated by breaking it into small Trotter steps, each applying local interaction terms sequentially.";
      const s = step.step ?? viewIndex;
      const fid = step.fidelity_vs_exact ?? step.fidelity;
      if (s === 0)
        return "Initial state: all probability concentrated in |00...0⟩. The Hamiltonian's interactions will redistribute this probability across other states as time evolves.";
      if (s <= 3)
        return `Trotter step ${s}: The interactions are causing probability to flow to other states. ${fid != null ? `Fidelity vs exact: ${(fid * 100).toFixed(1)}%` : ""} — each Trotter step introduces a small approximation error.`;
      if (s <= 8)
        return `Step ${s}: ${fid != null ? `Fidelity: ${(fid * 100).toFixed(1)}%.` : ""} The state distribution is evolving in a pattern characteristic of the model's interactions. More Trotter steps improve accuracy but deepen the circuit.`;
      return `Step ${s}: ${fid != null ? `Fidelity: ${(fid * 100).toFixed(2)}%.` : ""} The classical panel computes the same evolution using matrix exponentiation — notice the exponential cost O(2^n) in time and memory, while the quantum circuit uses only n qubits.`;
    },
    getDetails: (step) => ({
      "Trotter decomposition": "e^{-i(A+B)dt} ≈ e^{-iAdt} · e^{-iBdt} + O(dt²). First-order product formula.",
      "Error bound": "Trotter error per step: O(t²/r), where r = number of steps and t = total time.",
      "Ising model": "H = ∑ Jᵢⱼ ZᵢZⱼ + ∑ hᵢ Xᵢ. Transverse-field Ising model for magnetism.",
      "Classical cost": "Storing the state vector: 2^n complex numbers. Matrix expm: O(8^n) operations.",
      "Quantum cost": "O(n) qubits, O(n² · r) gates. Polynomial in system size.",
      "Fidelity": (step?.fidelity_vs_exact ?? step?.fidelity) != null
        ? `${((step.fidelity_vs_exact ?? step.fidelity) * 100).toFixed(3)}%`
        : "--",
    }),
  },
};

// ---------------------------------------------------------------------------
// Insights tab
// ---------------------------------------------------------------------------

function InsightsTab({ moduleId, currentStep, viewIndex }) {
  const content = MODULE_CONTENT[moduleId];
  if (!content) return null;
  const insight = content.getInsight(currentStep, viewIndex);
  const details = content.getDetails(currentStep || {});

  return (
    <>
      <div className="anno-section">
        <h4 className="anno-section-title">Problem</h4>
        <p className="anno-section-body">{content.problem}</p>
      </div>

      <div className="anno-section">
        <h4 className="anno-section-title">Algorithm Insight</h4>
        <div className="annotation-explanation">{insight}</div>
      </div>

      <div className="anno-section">
        <h4 className="anno-section-title">Technical Details</h4>
        <div className="detail-content">
          {Object.entries(details).map(([key, val]) => (
            <div key={key} className="detail-row">
              <span className="detail-key">{key}:</span>
              <code>{val}</code>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Fidelity tab — line chart of fidelity_vs_exact across Trotter steps
// ---------------------------------------------------------------------------

function FidelityTab({ quantumSteps, viewIndex }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const history = useMemo(() => {
    return quantumSteps
      .map((s, i) => ({
        step: s.step ?? i + 1,
        fidelity: s.fidelity_vs_exact ?? s.fidelity,
      }))
      .filter((d) => d.fidelity != null);
  }, [quantumSteps]);

  const visible = history.slice(0, Math.min(viewIndex + 1, history.length));
  const current = visible[visible.length - 1];

  useEffect(() => {
    if (!svgRef.current || visible.length === 0) return;

    const container = containerRef.current;
    const width = container?.clientWidth || 280;
    const height = 220;
    const margin = { top: 20, right: 18, bottom: 36, left: 48 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    svg.selectAll("*").remove();

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3
      .scaleLinear()
      .domain([1, Math.max(d3.max(history, (d) => d.step) || 1, 1)])
      .range([0, innerW]);

    const yMin = Math.min(d3.min(history, (d) => d.fidelity) ?? 1, 1) * 0.98;
    const yScale = d3.scaleLinear().domain([Math.max(yMin, 0), 1]).range([innerH, 0]);

    // Grid
    g.append("g")
      .selectAll("line")
      .data(yScale.ticks(4))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "#2a2a5a")
      .attr("stroke-opacity", 0.3);

    // Area under curve
    const area = d3
      .area()
      .x((d) => xScale(d.step))
      .y0(innerH)
      .y1((d) => yScale(d.fidelity))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(visible)
      .attr("d", area)
      .attr("fill", "rgba(0,212,255,0.12)");

    // Line
    const line = d3
      .line()
      .x((d) => xScale(d.step))
      .y((d) => yScale(d.fidelity))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(visible)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", "#00d4ff")
      .attr("stroke-width", 1.8);

    // Current marker
    if (current) {
      g.append("circle")
        .attr("cx", xScale(current.step))
        .attr("cy", yScale(current.fidelity))
        .attr("r", 4)
        .attr("fill", "#00d4ff")
        .attr("stroke", "#0a0a1a")
        .attr("stroke-width", 1.5);
    }

    // X axis
    const xAxis = g
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(Math.min(history.length, 6)).tickFormat(d3.format("d")));

    xAxis.selectAll("text").attr("font-size", "9px").attr("font-family", "'JetBrains Mono', monospace");

    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH + 28)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "10px")
      .text("Trotter Step");

    // Y axis
    g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".2%")));

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -36)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "10px")
      .text("Fidelity vs exact");
  }, [visible, history, current]);

  return (
    <>
      <div className="anno-section">
        <h4 className="anno-section-title">Trotter Fidelity</h4>
        <p className="anno-section-body">
          Fidelity between the Trotter-evolved state and the exact reference
          <code> e<sup>-iHt</sup></code>. Stays near 1.0 if <code>n_steps</code> is
          large enough; drifts down when Trotter error accumulates.
        </p>
      </div>

      <div className="anno-section">
        {history.length === 0 ? (
          <div className="placeholder" style={{ padding: 12, color: "var(--text-muted)" }}>
            Fidelity will appear once the race runs.
          </div>
        ) : (
          <div ref={containerRef} className="fidelity-chart">
            <svg ref={svgRef} />
            {current && (
              <div className="fidelity-current">
                Current: <code>{(current.fidelity * 100).toFixed(3)}%</code> at step <code>{current.step}</code>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Run Info tab — full metadata for both solvers
// ---------------------------------------------------------------------------

function formatVal(v) {
  if (v == null) return "--";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    return v.toFixed(4);
  }
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function MetadataBlock({ title, color, result }) {
  const md = result?.metadata;
  const fr = result?.final_result;
  return (
    <div className="anno-section">
      <h4 className="anno-section-title" style={{ color }}>{title}</h4>
      {!md && !fr ? (
        <div className="placeholder" style={{ padding: 8, color: "var(--text-muted)" }}>
          No data yet.
        </div>
      ) : (
        <div className="detail-content">
          {md && Object.entries(md).map(([k, v]) => (
            <div key={`md-${k}`} className="detail-row">
              <span className="detail-key">{k.replace(/_/g, " ")}:</span>
              <code>{formatVal(v)}</code>
            </div>
          ))}
          {fr && Object.entries(fr).filter(([k]) => k !== "measured_counts").map(([k, v]) => (
            <div key={`fr-${k}`} className="detail-row">
              <span className="detail-key">{k.replace(/_/g, " ")}:</span>
              <code>{formatVal(v)}</code>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RunInfoTab({ quantumResult, classicalResult }) {
  return (
    <>
      <MetadataBlock title="Quantum" color="var(--quantum-color)" result={quantumResult} />
      <MetadataBlock title="Classical" color="var(--classical-color)" result={classicalResult} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Tabbed shell
// ---------------------------------------------------------------------------

export default function Annotations({
  moduleId,
  quantumSteps,
  classicalSteps,
  quantumResult,
  classicalResult,
  viewIndex,
  raceState,
  activeTab,
  onTabClick,
}) {
  const showFidelity = moduleId === "hamiltonian_sim";
  const tabs = [
    { id: "insights", label: "Insights" },
    ...(showFidelity ? [{ id: "fidelity", label: "Fidelity" }] : []),
    { id: "runinfo", label: "Run Info" },
  ];

  // If active tab disappears (e.g. switch off hamiltonian_sim while on Fidelity), collapse.
  useEffect(() => {
    if (activeTab && !tabs.some((t) => t.id === activeTab)) onTabClick(activeTab);
  }, [moduleId]); // eslint-disable-line react-hooks/exhaustive-deps

  const currentStep = quantumSteps.length > 0
    ? quantumSteps[Math.min(viewIndex, quantumSteps.length - 1)]
    : null;

  return (
    <>
      <div className="anno-rail">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`anno-rail-btn ${activeTab === t.id ? "active" : ""}`}
            onClick={() => onTabClick(t.id)}
            title={t.label}
          >
            <span className="anno-rail-label">{t.label}</span>
          </button>
        ))}
      </div>
      {activeTab && (
        <div className="anno-content">
          {activeTab === "insights" && (
            <InsightsTab moduleId={moduleId} currentStep={currentStep} viewIndex={viewIndex} />
          )}
          {activeTab === "fidelity" && showFidelity && (
            <FidelityTab quantumSteps={quantumSteps} viewIndex={viewIndex} />
          )}
          {activeTab === "runinfo" && (
            <RunInfoTab quantumResult={quantumResult} classicalResult={classicalResult} />
          )}
        </div>
      )}
    </>
  );
}
