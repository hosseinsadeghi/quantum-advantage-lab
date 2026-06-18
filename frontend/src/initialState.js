/**
 * Generate initial-state (step 0) data for each module so the user
 * sees the problem setup before pressing Start Race.
 */

export function generateInitialState(moduleId, config) {
  switch (moduleId) {
    case "grovers_search":
      return groverSetup(config);
    case "quantum_walks":
      return walkSetup(config);
    case "vqe":
      return vqeSetup(config);
    case "hamiltonian_sim":
      return hamiltonianSetup(config);
    default:
      return { quantum: [], classical: [] };
  }
}

function groverSetup(config) {
  const n = config.n_qubits || 4;
  const N = Math.pow(2, n);
  const target = config.target_state ?? 7;

  // Uniform superposition: every state has amplitude 1/N
  const uniform = {};
  for (let i = 0; i < N; i++) {
    uniform[i.toString(2).padStart(n, "0")] = 1 / N;
  }

  const quantumStep = {
    iteration: 0,
    amplitudes: uniform,
    target_probability: 1 / N,
    description: "Initial uniform superposition — every state has equal probability 1/" + N,
  };

  // Classical: nothing checked yet — show empty grid with all items unchecked
  // We pass amplitudes as an all-zero array so the heatmap draws the full grid
  const classicalStep = {
    step: 0,
    amplitudes: new Array(N).fill(0),
    description: `Search space: ${N} items. Target: item ${target}. No items checked yet.`,
  };

  return {
    quantum: [quantumStep],
    classical: [classicalStep],
    // Provide result stubs so target marker is drawn on both sides
    quantumResult: { final_result: { target_state: target, search_space_size: N } },
    classicalResult: { final_result: { target: target, search_space_size: N } },
  };
}

function walkSetup(config) {
  const n = config.n_qubits || 4;
  const nNodes = Math.pow(2, n);
  const graphType = config.graph_type || "cycle";

  // Walker starts at node 0
  const qDist = {};
  for (let i = 0; i < nNodes; i++) {
    qDist[i] = i === 0 ? 1.0 : 0.0;
  }

  const cDist = new Array(nNodes).fill(0);
  cDist[0] = 1.0;

  const quantumStep = {
    step: 0,
    position_distribution: qDist,
    description: `Walker starts at node 0 on a ${graphType} graph with ${nNodes} nodes. Quantum interference will spread the walker faster.`,
  };

  const classicalStep = {
    step: 0,
    distribution: cDist,
    mean_position: 0,
    std_position: 0,
    description: `Walker starts at node 0. Classical random walk on a ${graphType} graph with ${nNodes} nodes.`,
  };

  return { quantum: [quantumStep], classical: [classicalStep] };
}

function vqeSetup(config) {
  const molecule = config.molecule || "H2";
  const nLayers = config.n_layers || 2;
  const exactEnergies = { H2: -1.8572750, LiH: -7.8825378 };
  const exact = exactEnergies[molecule] || -1.857;

  // Show a starting energy far from ground state
  const startEnergy = exact + 0.8 + Math.random() * 0.3;

  const quantumStep = {
    iteration: 0,
    energy: startEnergy,
    params: [],
    description: `Molecule: ${molecule}. Starting VQE with ${nLayers} ansatz layers. Target ground state: ${exact.toFixed(4)} Ha.`,
  };

  const classicalStep = {
    iteration: 0,
    energy: startEnergy + 0.1,
    params: [],
    converged: false,
    description: `Classical optimizer starting from random initial point for ${molecule}.`,
  };

  return {
    quantum: [quantumStep],
    classical: [classicalStep],
    // Provide result stub so the ground state line is drawn
    quantumResult: { final_result: { exact_ground_state_energy: exact } },
    classicalResult: { final_result: { exact_ground_state_energy: exact } },
  };
}

function hamiltonianSetup(config) {
  const n = config.n_qubits || 4;
  const N = Math.pow(2, n);
  const model = config.model || "ising";
  const pattern = config.interaction_pattern || "chain";

  // Initial state: all probability in |00...0⟩
  const probs = {};
  for (let i = 0; i < N; i++) {
    const label = i.toString(2).padStart(n, "0");
    probs[label] = i === 0 ? 1.0 : 0.0;
  }

  const quantumStep = {
    step: 0,
    time: 0,
    fidelity_vs_exact: 1.0,
    state_probabilities: probs,
    description: `Initial state |${"0".repeat(n)}⟩. ${model.charAt(0).toUpperCase() + model.slice(1)} model on ${n} qubits with ${pattern.replaceAll("_", " ")} couplings.`,
  };

  const classicalStep = {
    step: 0,
    time: 0,
    state_probs: { ...probs },
    wall_time_ms: 0,
    matrix_size: N,
    flops_estimate: 0,
    description: `Classical matrix exponentiation starting from |${"0".repeat(n)}⟩. Matrix size: ${N}×${N}.`,
  };

  return { quantum: [quantumStep], classical: [classicalStep] };
}
