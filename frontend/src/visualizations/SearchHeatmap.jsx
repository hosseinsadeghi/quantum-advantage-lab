import { useRef, useEffect, useMemo } from "react";
import * as d3 from "d3";

export default function SearchHeatmap({ steps, viewIndex, result, type }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const currentStep = steps[Math.min(viewIndex, steps.length - 1)];
  const isQuantum = type === "quantum";

  // --- Normalize amplitudes to a flat array ---
  // Quantum: dict {"0000": 0.0625, ...} → array of probabilities
  // Classical: {checked, found, progress} → array showing search progress
  const amplitudes = useMemo(() => {
    // Quantum side: has amplitudes dict or array
    const raw = currentStep?.amplitudes;
    if (raw) {
      if (Array.isArray(raw)) return raw;
      // Dict keyed by bitstring — convert to positional array
      const keys = Object.keys(raw).sort();
      const n = keys.length;
      const arr = new Array(n).fill(0);
      keys.forEach((k, idx) => { arr[idx] = raw[k]; });
      return arr;
    }

    // Classical side: reconstruct a grid showing what's been checked
    // Need total size from result or from steps context
    const totalItems = result?.final_result?.search_space_size
      || result?.metadata?.search_space_size
      || (steps.length > 0 ? steps.length : 0);

    if (totalItems === 0) return [];

    const arr = new Array(totalItems).fill(0);

    // Walk through all steps up to current viewIndex
    const upTo = Math.min(viewIndex, steps.length - 1);
    for (let i = 0; i <= upTo; i++) {
      const s = steps[i];
      const idx = s.checked ?? -1;
      if (idx >= 0 && idx < totalItems) {
        arr[idx] = s.found ? 1.0 : 0.4; // found = bright, checked = dim
      }
    }
    return arr;
  }, [currentStep, viewIndex, steps, result, isQuantum]);

  const target = result?.final_result?.target
    ?? result?.final_result?.target_state
    ?? result?.target
    ?? null;

  const dims = useMemo(() => {
    const n = amplitudes.length;
    if (n === 0) return { cols: 1, rows: 1 };
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    return { cols, rows };
  }, [amplitudes.length]);

  useEffect(() => {
    if (!svgRef.current || amplitudes.length === 0) return;

    const container = containerRef.current;
    const width = container?.clientWidth || 400;
    const height = container?.clientHeight || 280;
    const margin = { top: 24, right: 16, bottom: 8, left: 16 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const { cols, rows } = dims;
    const cellW = Math.min(innerW / cols, innerH / rows, 40);
    const cellH = cellW;
    const gridW = cols * cellW;
    const gridH = rows * cellH;
    const offsetX = margin.left + (innerW - gridW) / 2;
    const offsetY = margin.top + (innerH - gridH) / 2;

    const svg = d3.select(svgRef.current);
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    // Clear everything on each render — simpler and avoids stale elements
    svg.selectAll("*").remove();

    // Color scale
    const maxAmp = d3.max(amplitudes.map(Math.abs)) || 1;
    const colorScale = isQuantum
      ? d3.scaleSequential(d3.interpolateViridis).domain([0, maxAmp])
      : d3.scaleSequential(d3.interpolateOrRd).domain([0, maxAmp]);

    // Title
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 14)
      .attr("text-anchor", "middle")
      .attr("fill", "#9898b8")
      .attr("font-size", "11px")
      .attr("font-family", "Inter, sans-serif")
      .text(
        isQuantum
          ? `Quantum Amplitudes  |  Iteration ${currentStep?.iteration ?? viewIndex}`
          : `Classical Search  |  Step ${Math.min(viewIndex + 1, steps.length)}`
      );

    // Build cell data
    const data = amplitudes.map((amp, i) => ({
      i,
      amp: Math.abs(amp),
      col: i % cols,
      row: Math.floor(i / cols),
    }));

    // Draw cells
    svg
      .selectAll(".cell")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "cell")
      .attr("rx", 3)
      .attr("ry", 3)
      .attr("x", (d) => offsetX + d.col * cellW + 1)
      .attr("y", (d) => offsetY + d.row * cellH + 1)
      .attr("width", cellW - 2)
      .attr("height", cellH - 2)
      .attr("fill", (d) => d.amp > 0 ? colorScale(d.amp) : "rgba(30,30,63,0.6)")
      .attr("stroke", (d) =>
        target !== null && d.i === target
          ? (isQuantum ? "#00d4ff" : "#ff6b6b")
          : "none"
      )
      .attr("stroke-width", (d) =>
        target !== null && d.i === target ? 2 : 0
      );

    // State index labels (only if cells are big enough)
    if (cellW >= 20) {
      svg
        .selectAll(".cell-label")
        .data(data)
        .enter()
        .append("text")
        .attr("class", "cell-label")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("font-family", "'JetBrains Mono', monospace")
        .attr("font-size", Math.min(cellW * 0.3, 10) + "px")
        .attr("pointer-events", "none")
        .attr("x", (d) => offsetX + d.col * cellW + cellW / 2)
        .attr("y", (d) => offsetY + d.row * cellH + cellH / 2)
        .attr("fill", (d) => (d.amp > maxAmp * 0.6 ? "#0a0a1a" : "#9898b8"))
        .text((d) => `|${d.i}\u27E9`);
    }

    // Target marker (dashed outline)
    if (target !== null && target < amplitudes.length) {
      const tCol = target % cols;
      const tRow = Math.floor(target / cols);
      svg
        .append("rect")
        .attr("class", "target-marker")
        .attr("fill", "none")
        .attr("stroke", isQuantum ? "#00d4ff" : "#ff6b6b")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4,2")
        .attr("rx", 4)
        .attr("x", offsetX + tCol * cellW - 1)
        .attr("y", offsetY + tRow * cellH - 1)
        .attr("width", cellW + 2)
        .attr("height", cellH + 2);
    }
  }, [amplitudes, viewIndex, dims, target, isQuantum, steps.length, currentStep]);

  if (amplitudes.length === 0) {
    return <div className="placeholder">No data yet</div>;
  }

  return (
    <div className="viz-container" ref={containerRef}>
      <svg ref={svgRef} />
      <div className="viz-legend">
        {isQuantum ? (
          <>
            <div className="viz-legend-item">
              <div className="legend-swatch" style={{ background: "#440154" }} />
              <span>Low probability</span>
            </div>
            <div className="viz-legend-item">
              <div className="legend-swatch" style={{ background: "#fde725" }} />
              <span>High probability</span>
            </div>
          </>
        ) : (
          <>
            <div className="viz-legend-item">
              <div className="legend-swatch" style={{ background: "rgba(30,30,63,0.6)", border: "1px solid #2a2a5a" }} />
              <span>Not checked</span>
            </div>
            <div className="viz-legend-item">
              <div className="legend-swatch" style={{ background: "#fee5d9" }} />
              <span>Checked</span>
            </div>
            <div className="viz-legend-item">
              <div className="legend-swatch" style={{ background: "#de2d26" }} />
              <span>Found</span>
            </div>
          </>
        )}
        {target !== null && (
          <div className="viz-legend-item">
            <div
              className="legend-swatch"
              style={{
                background: "transparent",
                border: `1px dashed ${isQuantum ? "#00d4ff" : "#ff6b6b"}`,
                height: 8,
                width: 16,
              }}
            />
            <span>Target |{target}&#x27E9;</span>
          </div>
        )}
      </div>
    </div>
  );
}
