"""Base race module providing concurrent quantum/classical execution."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import os

from backend import cache as _cache

_executor = ThreadPoolExecutor(max_workers=4)


def _extract_counts(result: dict[str, Any]) -> dict[str, int]:
    """Best-effort counts extraction from a solver output dict."""
    fr = result.get("final_result") or {}
    for k in ("measured_counts", "counts", "samples"):
        v = fr.get(k)
        if isinstance(v, dict):
            return {str(bit): int(n) for bit, n in v.items()}
    return {}


@dataclass
class RaceResult:
    """Container for a completed race."""

    quantum_steps: list[dict[str, Any]] = field(default_factory=list)
    classical_steps: list[dict[str, Any]] = field(default_factory=list)
    quantum_result: dict[str, Any] = field(default_factory=dict)
    classical_result: dict[str, Any] = field(default_factory=dict)
    quantum_time: float = 0.0
    classical_time: float = 0.0


class RaceModule(ABC):
    """Base class for quantum-vs-classical race modules.

    Subclasses implement ``run_quantum`` and ``run_classical`` which call
    into the underlying solver packages.  The base class orchestrates
    concurrent execution and exposes both a blocking and a streaming
    interface.
    """

    # -- Metadata (override in subclasses) ------------------------------------
    module_id: str = ""
    title: str = ""
    description: str = ""
    default_params: dict[str, Any] = {}

    # -- Abstract solver hooks ------------------------------------------------

    @abstractmethod
    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run the quantum solver synchronously. Returns the solver dict."""
        ...

    @abstractmethod
    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run the classical solver synchronously. Returns the solver dict."""
        ...

    # -- Helpers --------------------------------------------------------------

    def _merged_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return default params with caller overrides applied."""
        merged = dict(self.default_params)
        if overrides:
            merged.update(overrides)
        return merged

    def _run_quantum_maybe_cached(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run ``run_quantum`` with cache consult + write on cacheable runs.

        Lookup is skipped when ``cache_bypass`` or ``add_shots`` is set, or
        when the resolved backend+size falls outside the cache policy.
        """
        backend = _cache.backend_id(params)
        n_qubits = int(params.get("n_qubits", 0) or 0)
        cacheable = _cache.should_cache(backend, n_qubits)
        # If the user targeted IonQ but IONQ_API_KEY is missing, the provider
        # silently falls back to Aer. Don't cache under the IonQ key — that
        # would return Aer results when IonQ is later available.
        if backend.startswith("ionq:") and not os.environ.get("IONQ_API_KEY"):
            cacheable = False
        bypass = bool(params.get("cache_bypass") or params.get("add_shots"))

        if cacheable and not bypass:
            key = _cache.compute_key(self.module_id, params)
            records = _cache.get(key)
            if records:
                agg = _cache.aggregate(records)
                md = dict(agg.get("metadata") or {})
                md["cache_hit"] = True
                md["cache_records"] = agg["n_records"]
                md["cache_shots"] = agg["shots"]
                return {
                    "steps": [],
                    "final_result": agg.get("final_result", {}),
                    "metadata": md,
                }

        result = self.run_quantum(params)

        if cacheable:
            key = _cache.compute_key(self.module_id, params)
            _cache.put(
                key,
                {
                    "shots": int(params.get("shots", 0) or 0),
                    "counts": _extract_counts(result),
                    "final_result": result.get("final_result", {}),
                    "metadata": result.get("metadata", {}),
                },
            )
        return result

    # -- Blocking interface ---------------------------------------------------

    async def run(self, params: dict[str, Any] | None = None) -> RaceResult:
        """Run both solvers concurrently and return the full result."""
        merged = self._merged_params(params)
        loop = asyncio.get_running_loop()

        t0 = time.perf_counter()

        quantum_future = loop.run_in_executor(
            _executor, self._run_quantum_maybe_cached, merged
        )
        classical_future = loop.run_in_executor(_executor, self.run_classical, merged)

        quantum_raw, classical_raw = await asyncio.gather(
            quantum_future, classical_future
        )

        elapsed = time.perf_counter() - t0

        result = RaceResult(
            quantum_steps=quantum_raw.get("steps", []),
            classical_steps=classical_raw.get("steps", []),
            quantum_result=quantum_raw.get("final_result", {}),
            classical_result=classical_raw.get("final_result", {}),
            quantum_time=quantum_raw.get("metadata", {}).get("elapsed", elapsed),
            classical_time=classical_raw.get("metadata", {}).get("elapsed", elapsed),
        )
        return result

    # -- Streaming interface --------------------------------------------------

    async def stream(
        self, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Run both solvers and yield step messages as they complete.

        Yields dicts with ``type`` in
        {``quantum_step``, ``classical_step``, ``complete``}.
        """
        merged = self._merged_params(params)
        loop = asyncio.get_running_loop()

        quantum_done = asyncio.Event()
        classical_done = asyncio.Event()

        quantum_raw: dict[str, Any] = {}
        classical_raw: dict[str, Any] = {}

        def _run_quantum() -> dict[str, Any]:
            return self._run_quantum_maybe_cached(merged)

        def _run_classical() -> dict[str, Any]:
            return self.run_classical(merged)

        quantum_future = loop.run_in_executor(_executor, _run_quantum)
        classical_future = loop.run_in_executor(_executor, _run_classical)

        # Track which futures have completed
        results: dict[str, dict[str, Any]] = {}

        def _on_quantum_done(fut: asyncio.Future) -> None:
            try:
                results["quantum"] = fut.result()
            except Exception as exc:
                results["quantum"] = {
                    "steps": [],
                    "final_result": {"error": str(exc)},
                    "metadata": {},
                }
            quantum_done.set()

        def _on_classical_done(fut: asyncio.Future) -> None:
            try:
                results["classical"] = fut.result()
            except Exception as exc:
                results["classical"] = {
                    "steps": [],
                    "final_result": {"error": str(exc)},
                    "metadata": {},
                }
            classical_done.set()

        quantum_future.add_done_callback(
            lambda f: loop.call_soon_threadsafe(lambda: _on_quantum_done(f))
        )
        classical_future.add_done_callback(
            lambda f: loop.call_soon_threadsafe(lambda: _on_classical_done(f))
        )

        # Wait for each future and yield steps as they complete
        pending = {"quantum", "classical"}
        while pending:
            done_events = []
            if "quantum" in pending:
                done_events.append(("quantum", quantum_done))
            if "classical" in pending:
                done_events.append(("classical", classical_done))

            # Wait for whichever finishes first
            wait_tasks = [
                asyncio.create_task(evt.wait()) for _, evt in done_events
            ]
            await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cancel remaining wait tasks
            for t in wait_tasks:
                t.cancel()

            # Yield steps for any newly-completed solver
            if "quantum" in pending and quantum_done.is_set():
                pending.discard("quantum")
                q = results["quantum"]
                for step in q.get("steps", []):
                    yield {"type": "quantum_step", "data": step}

            if "classical" in pending and classical_done.is_set():
                pending.discard("classical")
                c = results["classical"]
                for step in c.get("steps", []):
                    yield {"type": "classical_step", "data": step}

        q = results["quantum"]
        c = results["classical"]
        yield {
            "type": "complete",
            "data": {
                "quantum": {
                    "final_result": q.get("final_result", {}),
                    "metadata": q.get("metadata", {}),
                    "steps_count": len(q.get("steps", [])),
                },
                "classical": {
                    "final_result": c.get("final_result", {}),
                    "metadata": c.get("metadata", {}),
                    "steps_count": len(c.get("steps", [])),
                },
            },
        }

    # -- Serialisation helpers ------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Return module metadata suitable for the ``/api/modules`` listing."""
        return {
            "id": self.module_id,
            "title": self.title,
            "description": self.description,
            "default_params": self.default_params,
        }
