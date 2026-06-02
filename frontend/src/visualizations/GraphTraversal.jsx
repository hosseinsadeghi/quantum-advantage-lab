import { useRef, useEffect, useMemo } from "react";
import * as d3 from "d3";

export default function GraphTraversal({ steps, viewIndex, result, type }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const currentStep = steps[Math.min(viewIndex, steps.length - 1)];
  const isQuantum = type === "quantum";

  // Normalize distribution to array
  const distribution = useMemo(() => {
    const raw = currentStep?.distribution || currentStep?.position_distribution;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    const maxKey = Math.max(...Object.keys(raw).map(Number));
    const arr = new Array(maxKey + 1).fill(0);
    for (const [k, v] of Object.entries(raw)) arr[Number(k)] = v;
    return arr;
  }, [currentStep]);

  const nNodes = distribution.length;

  // Compute standard deviation for the spread indicator
  const stats = useMemo(() => {
    if (nNodes === 0) return { mean: 0, std: 0, spread: 0 };
    let mean = 0;
    for (let i = 0; i < nNodes; i++) mean += i * distribution[i];
    let variance = 0;
    for (let i = 0; i < nNodes; i++) variance += distribution[i] * (i - mean) ** 2;
    const std = Math.sqrt(variance);
    // "spread" = fraction of nodes with non-negligible probability (> 1%)
    const spread = distribution.filter((p) => p > 0.01).length / nNodes;
    return { mean, std, spread };
  }, [distribution, nNodes]);

  // Step index
  const stepNum = currentStep?.step ?? viewIndex;

  useEffect(() => {
    if (!svgRef.current || nNodes === 0) return;

    const container = containerRef.current;
    const width = container?.clientWidth || 400;
    const height = container?.clientHeight || 280;
    const margin = { top: 30, right: 16, bottom: 44, left: 44 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    svg.selectAll("*").remove();

    const barColor = isQuantum ? "#00d4ff" : "#ff6b6b";
    const barDim = isQuantum ? "rgba(0,212,255,0.25)" : "rgba(255,107,107,0.25)";
    const accentColor = isQuantum ? "#4361ee" : "#cc4444";

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
        `${isQuantum ? "Quantum" : "Classical"} Walk  |  Step ${stepNum}  |  Spread: ${(stats.spread * 100).toFixed(0)}% of nodes`
      );

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Uniform reference line (1/N)
    const uniformProb = 1 / nNodes;

    // Scales — fix y-domain to make comparison fair
    const xScale = d3
      .scaleBand()
      .domain(d3.range(nNodes))
      .range([0, innerW])
      .padding(nNodes > 16 ? 0.1 : 0.2);

    const maxProb = Math.max(d3.max(distribution) || 0, uniformProb * 2, 0.05);
    const yScale = d3.scaleLinear().domain([0, maxProb * 1.1]).range([innerH, 0]);

    // Grid lines
    g.selectAll(".grid-line")
      .data(yScale.ticks(4))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "#2a2a5a")
      .attr("stroke-opacity", 0.3);

    // Uniform reference line
    g.append("line")
      .attr("x1", 0)
      .attr("x2", innerW)
      .attr("y1", yScale(uniformProb))
      .attr("y2", yScale(uniformProb))
      .attr("stroke", "#6868a0")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,3")
      .attr("opacity", 0.6);

    g.append("text")
      .attr("x", innerW - 2)
      .attr("y", yScale(uniformProb) - 5)
      .attr("text-anchor", "end")
      .attr("fill", "#6868a0")
      .attr("font-size", "8px")
      .attr("font-family", "'JetBrains Mono', monospace")
      .text("uniform 1/N");

    // Bars
    g.selectAll(".prob-bar")
      .data(distribution)
      .enter()
      .append("rect")
      .attr("class", "prob-bar")
      .attr("x", (_, i) => xScale(i))
      .attr("width", xScale.bandwidth())
      .attr("y", (d) => yScale(d))
      .attr("height", (d) => innerH - yScale(d))
      .attr("rx", Math.min(xScale.bandwidth() / 4, 3))
      .attr("fill", (d) => (d > uniformProb ? barColor : barDim));

    // Start position marker (node 0)
    const startX = xScale(0) + xScale.bandwidth() / 2;
    g.append("line")
      .attr("x1", startX)
      .attr("x2", startX)
      .attr("y1", innerH + 2)
      .attr("y2", innerH + 10)
      .attr("stroke", "#22c55e")
      .attr("stroke-width", 2);
    g.append("text")
      .attr("x", startX)
      .attr("y", innerH + 19)
      .attr("text-anchor", "middle")
      .attr("fill", "#22c55e")
      .attr("font-size", "7px")
      .attr("font-family", "'JetBrains Mono', monospace")
      .text("start");

    // Standard deviation bracket
    if (stats.std > 0.5) {
      const left = Math.max(0, Math.floor(stats.mean - stats.std));
      const right = Math.min(nNodes - 1, Math.ceil(stats.mean + stats.std));
      const bracketY = yScale(maxProb * 1.03);
      const x1 = xScale(left);
      const x2 = xScale(right) + xScale.bandwidth();

      // Bracket line
      g.append("line")
        .attr("x1", x1).attr("x2", x2)
        .attr("y1", bracketY).attr("y2", bracketY)
        .attr("stroke", barColor).attr("stroke-width", 1.5).attr("opacity", 0.7);
      // Left tick
      g.append("line")
        .attr("x1", x1).attr("x2", x1)
        .attr("y1", bracketY - 4).attr("y2", bracketY + 4)
        .attr("stroke", barColor).attr("stroke-width", 1.5).attr("opacity", 0.7);
      // Right tick
      g.append("line")
        .attr("x1", x2).attr("x2", x2)
        .attr("y1", bracketY - 4).attr("y2", bracketY + 4)
        .attr("stroke", barColor).attr("stroke-width", 1.5).attr("opacity", 0.7);
      // Label
      g.append("text")
        .attr("x", (x1 + x2) / 2)
        .attr("y", bracketY - 6)
        .attr("text-anchor", "middle")
        .attr("fill", barColor)
        .attr("font-size", "8px")
        .attr("font-family", "'JetBrains Mono', monospace")
        .text(`\u03C3 = ${stats.std.toFixed(2)}`);
    }

    // X axis
    const showEveryN = nNodes > 16 ? Math.ceil(nNodes / 8) : 1;
    g.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerH})`)
      .call(
        d3.axisBottom(xScale)
          .tickSize(3)
          .tickValues(d3.range(nNodes).filter((i) => i % showEveryN === 0))
      )
      .selectAll("text")
      .attr("font-size", "8px")
      .attr("font-family", "'JetBrains Mono', monospace");

    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH + 34)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "9px")
      .text("Node position");

    // Y axis
    g.append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(yScale).ticks(4).tickSize(3).tickFormat(d3.format(".2f")));

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -32)
      .attr("text-anchor", "middle")
      .attr("fill", "#6868a0")
      .attr("font-size", "9px")
      .text("Probability");
  }, [distribution, nNodes, isQuantum, stats, stepNum]);

  if (nNodes === 0) {
    return <div className="placeholder">No distribution data</div>;
  }

  return (
    <div className="viz-container" ref={containerRef}>
      <svg ref={svgRef} />
      <div className="viz-legend">
        <div className="viz-legend-item">
          <div
            className="legend-swatch"
            style={{ background: isQuantum ? "#00d4ff" : "#ff6b6b", height: 8, width: 12, borderRadius: 2 }}
          />
          <span>P &gt; 1/N</span>
        </div>
        <div className="viz-legend-item">
          <div
            className="legend-swatch"
            style={{ background: isQuantum ? "rgba(0,212,255,0.25)" : "rgba(255,107,107,0.25)", height: 8, width: 12, borderRadius: 2 }}
          />
          <span>P &le; 1/N</span>
        </div>
        <div className="viz-legend-item">
          <div className="legend-swatch" style={{ background: "#22c55e", height: 3, width: 12 }} />
          <span>Start</span>
        </div>
        <div className="viz-legend-item">
          <div className="legend-swatch" style={{ borderTop: "1px dashed #6868a0", height: 1, width: 12 }} />
          <span>Uniform 1/N</span>
        </div>
      </div>
    </div>
  );
}
