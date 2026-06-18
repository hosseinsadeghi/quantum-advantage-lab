import { useRef, useEffect } from "react";
import * as d3 from "d3";

export default function EnergyLandscape({ steps, viewIndex, result, type, sharedScale }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const isQuantum = type === "quantum";

  // Extract energy series up to current view
  const visibleSteps = steps.slice(0, Math.min(viewIndex + 1, steps.length));
  const energies = visibleSteps.map((s, i) => ({
    iteration: s.iteration ?? i,
    energy: s.energy ?? s.value ?? 0,
  }));

  const groundStateEnergy = result?.final_result?.exact_ground_state_energy ?? result?.final_result?.ground_state_energy ?? null;
  const chemAccuracy = result?.final_result?.chemical_accuracy ?? 0.0016; // 1.6 mHa

  useEffect(() => {
    if (!svgRef.current || energies.length === 0) return;

    const container = containerRef.current;
    const width = container?.clientWidth || 400;
    const height = 280;
    const margin = { top: 28, right: 24, bottom: 40, left: 56 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    // Use shared scale if available (both panels use same axes), otherwise compute locally
    let xDomain, yDomain;
    if (sharedScale) {
      xDomain = [0, sharedScale.xMax];
      yDomain = [sharedScale.yMin, sharedScale.yMax];
    } else {
      const allEnergies = steps.map((s) => s.energy ?? s.value ?? 0);
      const allIterations = steps.map((s, i) => s.iteration ?? i);
      xDomain = [0, d3.max(allIterations) || energies.length];
      const yMin = d3.min([...allEnergies, groundStateEnergy].filter((v) => v != null)) || -2;
      const yMax = d3.max(allEnergies) || 0;
      const yPadding = (yMax - yMin) * 0.15;
      yDomain = [yMin - yPadding, yMax + yPadding];
    }

    const xScale = d3.scaleLinear().domain(xDomain).range([0, innerW]);
    const yScale = d3.scaleLinear().domain(yDomain).range([innerH, 0]);

    // Clear and redraw
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
      .text(`${isQuantum ? "VQE" : "Classical Optimizer"}  |  Iteration ${energies.length}`);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Grid
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(yScale.ticks(5))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "#2a2a5a")
      .attr("stroke-opacity", 0.3);

    // Ground state energy line
    if (groundStateEnergy != null) {
      g.append("line")
        .attr("x1", 0)
        .attr("x2", innerW)
        .attr("y1", yScale(groundStateEnergy))
        .attr("y2", yScale(groundStateEnergy))
        .attr("stroke", "#22c55e")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "6,4")
        .attr("opacity", 0.8);

      g.append("text")
        .attr("x", innerW - 4)
        .attr("y", yScale(groundStateEnergy) - 6)
        .attr("text-anchor", "end")
        .attr("fill", "#22c55e")
        .attr("font-size", "9px")
        .attr("font-family", "'JetBrains Mono', monospace")
        .text(`Exact: ${groundStateEnergy.toFixed(4)} Ha`);

      // Chemical accuracy band
      if (chemAccuracy) {
        g.append("rect")
          .attr("x", 0)
          .attr("y", yScale(groundStateEnergy + chemAccuracy))
          .attr("width", innerW)
          .attr("height", Math.abs(yScale(groundStateEnergy - chemAccuracy) - yScale(groundStateEnergy + chemAccuracy)))
          .attr("fill", "#22c55e")
          .attr("opacity", 0.07)
          .attr("rx", 2);
      }
    }

    // Line generator
    const line = d3
      .line()
      .x((d) => xScale(d.iteration))
      .y((d) => yScale(d.energy))
      .curve(d3.curveMonotoneX);

    // Energy path
    const lineColor = isQuantum ? "#00d4ff" : "#ff6b6b";

    const path = g
      .append("path")
      .datum(energies)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", lineColor)
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round");

    // Animate line drawing
    const pathLength = path.node().getTotalLength();
    path
      .attr("stroke-dasharray", pathLength)
      .attr("stroke-dashoffset", pathLength)
      .transition()
      .duration(400)
      .ease(d3.easeLinear)
      .attr("stroke-dashoffset", 0);

    // Area under the curve
    const area = d3
      .area()
      .x((d) => xScale(d.iteration))
      .y0(innerH)
      .y1((d) => yScale(d.energy))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(energies)
      .attr("d", area)
      .attr("fill", lineColor)
      .attr("opacity", 0.06);

    // Data points
    g.selectAll(".energy-dot")
      .data(energies)
      .enter()
      .append("circle")
      .attr("class", "energy-dot")
      .attr("cx", (d) => xScale(d.iteration))
      .attr("cy", (d) => yScale(d.energy))
      .attr("r", energies.length > 30 ? 2 : 3.5)
      .attr("fill", lineColor)
      .attr("stroke", "#0a0a1a")
      .attr("stroke-width", 1)
      .attr("opacity", 0)
      .transition()
      .delay((_, i) => i * 15)
      .duration(200)
      .attr("opacity", 1);

    // Current point highlight
    if (energies.length > 0) {
      const last = energies[energies.length - 1];
      g.append("circle")
        .attr("cx", xScale(last.iteration))
        .attr("cy", yScale(last.energy))
        .attr("r", 5)
        .attr("fill", lineColor)
        .attr("stroke", "#0a0a1a")
        .attr("stroke-width", 2)
        .attr("filter", `drop-shadow(0 0 4px ${lineColor})`);

      // Energy label
      g.append("text")
        .attr("x", xScale(last.iteration) + 8)
        .attr("y", yScale(last.energy) - 8)
        .attr("fill", lineColor)
        .attr("font-size", "10px")
        .attr("font-family", "'JetBrains Mono', monospace")
        .text(`${last.energy.toFixed(4)}`);
    }

    // X axis
    g.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSize(4));

    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH + 32)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "10px")
      .text("Iteration");

    // Y axis
    g.append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(yScale).ticks(5).tickSize(4).tickFormat(d3.format(".3f")));

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -44)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "10px")
      .text("Energy (Ha)");
  }, [energies, groundStateEnergy, chemAccuracy, isQuantum, steps, sharedScale]);

  if (energies.length === 0) {
    return <div className="placeholder">Waiting for energy data...</div>;
  }

  return (
    <div className="viz-container" ref={containerRef}>
      <svg ref={svgRef} />
      <div className="viz-legend">
        <div className="viz-legend-item">
          <div
            className="legend-swatch"
            style={{ background: isQuantum ? "#00d4ff" : "#ff6b6b" }}
          />
          <span>{isQuantum ? "VQE Energy" : "Classical Energy"}</span>
        </div>
        {groundStateEnergy != null && (
          <>
            <div className="viz-legend-item">
              <div
                className="legend-swatch"
                style={{ background: "#22c55e", height: 2, borderTop: "1px dashed #22c55e" }}
              />
              <span>Exact ground state</span>
            </div>
            <div className="viz-legend-item">
              <div
                className="legend-swatch"
                style={{ background: "rgba(34,197,94,0.15)", height: 8, width: 16 }}
              />
              <span>Chemical accuracy</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
