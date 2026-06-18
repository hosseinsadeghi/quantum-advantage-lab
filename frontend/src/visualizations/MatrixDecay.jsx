import { useRef, useEffect, useMemo } from "react";
import * as d3 from "d3";

export default function MatrixDecay({ steps, viewIndex, result, type, sharedScale }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const isQuantum = type === "quantum";
  const currentStep = steps[Math.min(viewIndex, steps.length - 1)];

  // Both backends send state probs as dicts {"0000": 0.95, ...}
  // Quantum uses state_probabilities, classical uses state_probs
  const stateProbs = useMemo(() => {
    const raw = currentStep?.state_probabilities || currentStep?.state_probs || currentStep?.distribution;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    // Dict with bitstring keys: convert to sorted array
    const keys = Object.keys(raw).sort();
    return keys.map(k => raw[k]);
  }, [currentStep]);

  useEffect(() => {
    if (!svgRef.current || stateProbs.length === 0) return;

    const container = containerRef.current;
    const width = container?.clientWidth || 400;
    const height = 300;
    const margin = { top: 28, right: 24, bottom: 44, left: 48 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    svg.selectAll("*").remove();

    // Title
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .attr("fill", "#9898b8")
      .attr("font-size", "11px")
      .attr("font-family", "Inter, sans-serif")
      .text(
        `${isQuantum ? "Quantum" : "Classical"} State Distribution  |  Trotter Step ${Math.min(viewIndex + 1, steps.length)}`
      );

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const barColor = isQuantum ? "#4361ee" : "#ff6b6b";
    const barHighlight = isQuantum ? "#00d4ff" : "#ff9a9a";

    // --- Bar chart (full panel; both panels share the same Y scale) ---
    const nStates = sharedScale?.nStates || stateProbs.length;
    const labels = Array.from({ length: nStates }, (_, i) => `|${i}⟩`);

    const xScale = d3
      .scaleBand()
      .domain(labels)
      .range([0, innerW])
      .padding(0.2);

    const yMax = sharedScale?.yMax ?? Math.max((d3.max(stateProbs) || 0) * 1.15, 0.1);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    // Grid lines
    g.append("g")
      .attr("class", "grid")
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

    const localMax = d3.max(stateProbs) || 0;

    // Bars (only render bars for indices that exist in stateProbs)
    const bars = g
      .selectAll(".state-bar")
      .data(stateProbs)
      .enter()
      .append("rect")
      .attr("class", "state-bar")
      .attr("x", (_, i) => xScale(labels[i]))
      .attr("width", xScale.bandwidth())
      .attr("y", innerH)
      .attr("height", 0)
      .attr("rx", 2)
      .attr("fill", (d) => (d === localMax && localMax > 0 ? barHighlight : barColor))
      .attr("opacity", 0.85);

    bars
      .transition()
      .duration(400)
      .ease(d3.easeCubicOut)
      .attr("y", (d) => yScale(d))
      .attr("height", (d) => innerH - yScale(d));

    // Bar value labels (if not too many)
    if (nStates <= 16) {
      g.selectAll(".bar-label")
        .data(stateProbs)
        .enter()
        .append("text")
        .attr("class", "bar-label")
        .attr("x", (_, i) => xScale(labels[i]) + xScale.bandwidth() / 2)
        .attr("y", (d) => yScale(d) - 4)
        .attr("text-anchor", "middle")
        .attr("fill", "#9898b8")
        .attr("font-size", "8px")
        .attr("font-family", "'JetBrains Mono', monospace")
        .text((d) => (d > 0.005 ? d.toFixed(3) : ""));
    }

    // X axis
    const xAxis = g
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerH})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickSize(4)
          .tickValues(nStates <= 16 ? labels : labels.filter((_, i) => i % Math.ceil(nStates / 8) === 0))
      );

    xAxis
      .selectAll("text")
      .attr("font-size", "8px")
      .attr("font-family", "'JetBrains Mono', monospace");

    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH + 32)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "10px")
      .text("Basis State");

    // Y axis
    g.append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(yScale).ticks(4).tickSize(4).tickFormat(d3.format(".2f")));

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -36)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "10px")
      .text("Probability");
  }, [stateProbs, viewIndex, isQuantum, steps.length, sharedScale]);

  if (stateProbs.length === 0) {
    return <div className="placeholder">Waiting for state data...</div>;
  }

  return (
    <div className="viz-container" ref={containerRef}>
      <svg ref={svgRef} />
      <div className="viz-legend">
        <div className="viz-legend-item">
          <div
            className="legend-swatch"
            style={{ background: isQuantum ? "#4361ee" : "#ff6b6b", height: 8, width: 12, borderRadius: 2 }}
          />
          <span>State probability</span>
        </div>
        <div className="viz-legend-item">
          <div
            className="legend-swatch"
            style={{ background: isQuantum ? "#00d4ff" : "#ff9a9a", height: 8, width: 12, borderRadius: 2 }}
          />
          <span>Peak state</span>
        </div>
      </div>
    </div>
  );
}
