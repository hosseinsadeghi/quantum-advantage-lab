"""Classical Linear Search.

Brute-force linear search counterpart to Grover's algorithm.
Checks each item sequentially until the target is found.
"""

from __future__ import annotations

import time
from typing import Any


def run_linear_search(
    n_items: int,
    target: int,
) -> dict[str, Any]:
    """Execute a brute-force linear search over *n_items* looking for *target*.

    This is the classical counterpart to Grover's quantum search.  While
    Grover's finds the target in O(sqrt(N)) queries, classical search
    requires O(N) queries on average.

    Parameters
    ----------
    n_items:
        Total number of items in the search space.
    target:
        The index of the item to find (0 <= target < n_items).

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Each step records which item was checked and whether the target
            was found.
        final_result : dict
            Summary of the search outcome.
        metadata : dict
            Timing and algorithmic information.
    """
    if target < 0 or target >= n_items:
        raise ValueError(
            f"target {target} out of range for search space of size {n_items}"
        )

    steps: list[dict[str, Any]] = []
    found = False
    found_at_step = -1

    wall_start = time.perf_counter()

    for i in range(n_items):
        step_start = time.perf_counter()

        is_target = i == target
        progress = (i + 1) / n_items

        step_end = time.perf_counter()
        step_record: dict[str, Any] = {
            "step": i + 1,
            "checked": i,
            "found": is_target,
            "progress": progress,
            "wall_time_ms": (step_end - step_start) * 1000,
        }
        steps.append(step_record)

        if is_target:
            found = True
            found_at_step = i + 1
            break

    wall_end = time.perf_counter()
    total_wall_ms = (wall_end - wall_start) * 1000

    return {
        "steps": steps,
        "final_result": {
            "found": found,
            "target": target,
            "found_at_step": found_at_step,
            "total_checks": found_at_step if found else n_items,
            "search_space_size": n_items,
            "success_probability": 1.0 if found else 0.0,
        },
        "metadata": {
            "algorithm": "linear_search",
            "search_space_size": n_items,
            "target": target,
            "total_wall_time_ms": total_wall_ms,
            "complexity": "O(N)",
            "average_case_checks": n_items / 2,
            "worst_case_checks": n_items,
        },
    }
