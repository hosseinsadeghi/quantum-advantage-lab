"""Classical Random Walk.

Monte Carlo simulation of classical random walks on graphs, serving as
the counterpart to discrete-time quantum walks.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _build_adjacency(n_nodes: int, graph_type: str) -> np.ndarray:
    """Return the adjacency matrix for the specified graph type.

    Parameters
    ----------
    n_nodes:
        Number of nodes in the graph.
    graph_type:
        ``"cycle"`` or ``"complete"``.
    """
    if graph_type == "cycle":
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            adj[i, (i + 1) % n_nodes] = 1.0
            adj[i, (i - 1) % n_nodes] = 1.0
        return adj
    elif graph_type == "complete":
        adj = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        return adj
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type!r}")


def _transition_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Compute the row-stochastic transition matrix from an adjacency matrix."""
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolated nodes
    return adjacency / row_sums


def run_classical_walk(
    n_nodes: int,
    n_steps: int,
    graph_type: str = "cycle",
    n_trials: int = 1000,
) -> dict[str, Any]:
    """Execute a classical random walk via Monte Carlo simulation.

    This is the classical counterpart to the quantum walk.  A classical
    walker starts at node 0 and moves to a uniformly random neighbour at
    each step.  The simulation is repeated *n_trials* times to build a
    probability distribution over positions.

    Parameters
    ----------
    n_nodes:
        Number of nodes in the graph.
    n_steps:
        Number of walk steps.
    graph_type:
        ``"cycle"`` or ``"complete"``.
    n_trials:
        Number of independent random walk trials for the Monte Carlo
        estimate.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Per-step position probability distribution and statistics.
        final_result : dict
            Final distribution and dominant position.
        metadata : dict
            Timing and algorithmic information.
    """
    graph_type = graph_type.lower()
    if graph_type not in ("cycle", "complete"):
        raise ValueError(f"Unsupported graph_type: {graph_type!r}")

    adjacency = _build_adjacency(n_nodes, graph_type)
    transition = _transition_matrix(adjacency)

    rng = np.random.default_rng(42)

    wall_start = time.perf_counter()

    # Run all trials at once for efficiency: positions shape (n_trials,)
    positions = np.zeros(n_trials, dtype=int)  # all walkers start at node 0

    steps: list[dict[str, Any]] = []

    # Record initial distribution
    initial_dist = np.zeros(n_nodes)
    initial_dist[0] = 1.0
    steps.append({
        "step": 0,
        "distribution": initial_dist.tolist(),
        "mean_position": 0.0,
        "std_position": 0.0,
        "description": "Initial state (all walkers at node 0)",
    })

    for step_idx in range(1, n_steps + 1):
        step_start = time.perf_counter()

        # For each walker, pick next position based on transition probabilities
        new_positions = np.empty(n_trials, dtype=int)
        for trial_idx in range(n_trials):
            current = positions[trial_idx]
            new_positions[trial_idx] = rng.choice(
                n_nodes, p=transition[current]
            )
        positions = new_positions

        # Build empirical distribution
        counts = np.bincount(positions, minlength=n_nodes)
        distribution = (counts / n_trials).tolist()

        mean_pos = float(np.mean(positions))
        std_pos = float(np.std(positions))

        step_end = time.perf_counter()

        steps.append({
            "step": step_idx,
            "distribution": distribution,
            "mean_position": mean_pos,
            "std_position": std_pos,
            "wall_time_ms": (step_end - step_start) * 1000,
            "description": f"After step {step_idx}",
        })

    wall_end = time.perf_counter()
    total_wall_ms = (wall_end - wall_start) * 1000

    # Final distribution
    final_dist = steps[-1]["distribution"]
    dominant_position = int(np.argmax(final_dist))

    return {
        "steps": steps,
        "final_result": {
            "final_distribution": final_dist,
            "dominant_position": dominant_position,
            "n_steps": n_steps,
            "graph_type": graph_type,
            "n_nodes": n_nodes,
        },
        "metadata": {
            "algorithm": "classical_random_walk",
            "graph_type": graph_type,
            "n_nodes": n_nodes,
            "n_steps": n_steps,
            "n_trials": n_trials,
            "total_wall_time_ms": total_wall_ms,
            "complexity": "O(n_steps * n_trials)",
            "spreading_rate": "O(sqrt(t))",
        },
    }
