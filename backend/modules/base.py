"""Base race module providing concurrent quantum/classical execution."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

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


def _sanitized_final_result_for_cache(final_result: dict[str, Any]) -> dict[str, Any]:
    """Strip non-essential API-key metadata before persisting cache records."""
    sanitized = dict(final_result or {})
    execution = sanitized.get("execution")
    if isinstance(execution, dict):
        execution = dict(execution)
        execution.pop("key_name", None)
        message = execution.get("message")
        if isinstance(message, str) and "api key" in message.lower():
            execution.pop("message", None)
        sanitized["execution"] = execution
    return sanitized


def _fell_back(result: dict[str, Any]) -> bool:
    """True if the run silently fell back to Aer instead of the requested
    backend. Read from the provider's ``execution`` info that every quantum
    module surfaces under ``final_result`` (``{"actual", "fell_back", ...}``).
    Absent info → treat as no fallback (e.g. a genuine Aer run)."""
    execution = (result.get("final_result") or {}).get("execution") or {}
    return bool(execution.get("fell_back"))


def _result_matches_requested_backend(
    backend: str,
    params: dict[str, Any],
    final_result: dict[str, Any],
) -> bool:
    """True when *final_result* proves it came from the requested backend."""
    execution = final_result.get("execution") or {}
    if not isinstance(execution, dict):
        execution = {}

    if backend.startswith("ionq:qpu:"):
        requested = params.get("qpu_name", "qpu.forte-1")
        return (
            execution.get("requested") == requested
            and execution.get("actual") == requested
            and execution.get("fell_back") is False
        )

    if backend.startswith("ionq:emulator:"):
        return (
            execution.get("requested") == "ionq_simulator"
            and execution.get("actual") == "ionq_simulator"
            and execution.get("fell_back") is False
        )

    return not bool(execution.get("fell_back"))


def _cache_records_for_lookup(
    backend: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only records that are safe to replay for *params*."""
    if not backend.startswith("ionq:"):
        return records
    return [
        r for r in records
        if _result_matches_requested_backend(
            backend,
            params,
            (r.get("final_result") or {}),
        )
    ]


def _with_cached_hamiltonian_final_samples(
    module_id: str,
    steps: list[dict[str, Any]],
    final_result: dict[str, Any],
    counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """For cached Hamiltonian runs, show measured samples at the final snapshot."""
    if module_id != "hamiltonian_sim" or not steps or not counts:
        return steps, final_result

    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return steps, final_result

    final_probs = {str(bit): int(n) / total for bit, n in sorted(counts.items())}
    replay_steps = [dict(step) for step in steps]
    final_step = dict(replay_steps[-1])
    final_step["state_probabilities"] = final_probs
    final_step["sampled_counts"] = {str(bit): int(n) for bit, n in sorted(counts.items())}
    final_step["sampled_shots"] = total
    final_step["description"] = (
        f"{final_step.get('description', 'Final Trotter step')} "
        f"(cached measured samples, {total} shots)"
    )
    replay_steps[-1] = final_step

    replay_final = dict(final_result or {})
    replay_final["measured_counts"] = {str(bit): int(n) for bit, n in sorted(counts.items())}
    return replay_steps, replay_final


@dataclass
class RaceResult:
    """Container for a completed race."""

    quantum_steps: list[dict[str, Any]] = field(default_factory=list)
    classical_steps: list[dict[str, Any]] = field(default_factory=list)
    quantum_result: dict[str, Any] = field(default_factory=dict)
    classical_result: dict[str, Any] = field(default_factory=dict)
    quantum_metadata: dict[str, Any] = field(default_factory=dict)
    classical_metadata: dict[str, Any] = field(default_factory=dict)
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
        bypass = bool(params.get("cache_bypass") or params.get("add_shots"))

        # Lookup never depends on credentials: a cached record is portable
        # hardware data — anyone holding the repo-backed .cache/runs/*.jsonl
        # can replay it, no IonQ key required. The API key is not part of the
        # cache key.
        if cacheable and not bypass:
            key, raw_records = _cache.lookup(self.module_id, params)
            records = _cache_records_for_lookup(
                backend,
                params,
                raw_records,
            )
            if records:
                agg = _cache.aggregate(records)
                md = dict(agg.get("metadata") or {})
                md["cache_hit"] = True
                md["cache_records"] = agg["n_records"]
                md["cache_shots"] = agg["shots"]
                md["cache_key"] = key
                steps, final_result = _with_cached_hamiltonian_final_samples(
                    self.module_id,
                    agg.get("steps", []),
                    agg.get("final_result", {}),
                    agg.get("counts", {}),
                )
                # Replay the stored per-iteration steps so the UI animates the
                # cached run exactly like a fresh one. Without them the quantum
                # panel has nothing to draw and stays on "Waiting for race…".
                return {
                    "steps": steps,
                    "final_result": final_result,
                    "metadata": md,
                }

        result = self.run_quantum(params)

        # Only persist a result that actually ran on the requested backend. If
        # the provider silently fell back to Aer (missing key/package, IonQ
        # outage), writing it under the ``ionq:*`` key would later serve Aer
        # data as if it were hardware. Detect the fallback from what *executed*
        # (``execution.fell_back``), not from the local key env — so the cache
        # stays correct and portable regardless of who holds which credentials.
        if cacheable and _result_matches_requested_backend(
            backend,
            params,
            result.get("final_result") or {},
        ):
            key = _cache.compute_key(self.module_id, params)
            counts = _extract_counts(result)
            # Record the shots actually executed, not params["shots"] — the shot
            # count is resolved at the module boundary (resolve_shots) and never
            # written back to params, so reading it from params here yields 0.
            # Counts are ground truth; fall back to the solver's reported shots.
            shots_used = (
                sum(counts.values())
                or int((result.get("metadata") or {}).get("shots") or 0)
                or int(params.get("shots", 0) or 0)
            )
            _cache.put(
                key,
                {
                    "shots": int(shots_used),
                    "counts": counts,
                    "steps": result.get("steps", []),
                    "final_result": _sanitized_final_result_for_cache(
                        result.get("final_result", {})
                    ),
                    "metadata": result.get("metadata", {}),
                },
            )
        return result

    def cache_status(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return pre-run cache status for the merged module parameters."""
        merged = self._merged_params(params)
        backend = _cache.backend_id(merged)
        n_qubits = int(merged.get("n_qubits", 0) or 0)
        cacheable = _cache.should_cache(backend, n_qubits)
        current_key = _cache.compute_key(self.module_id, merged)

        if not cacheable:
            return {
                "cacheable": False,
                "has_cached_result": False,
                "backend": backend,
                "n_qubits": n_qubits,
                "records": 0,
                "shots": 0,
                "key": current_key,
            }

        matched_key, raw_records = _cache.lookup(self.module_id, merged)
        records = _cache_records_for_lookup(
            backend,
            merged,
            raw_records,
        )
        agg = _cache.aggregate(records) if records else {}
        return {
            "cacheable": True,
            "has_cached_result": bool(records),
            "backend": backend,
            "n_qubits": n_qubits,
            "records": int(agg.get("n_records", 0) or 0),
            "shots": int(agg.get("shots", 0) or 0),
            "key": current_key,
            "matched_key": matched_key if records else None,
        }

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
            quantum_metadata=quantum_raw.get("metadata", {}),
            classical_metadata=classical_raw.get("metadata", {}),
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
        errors: dict[str, BaseException] = {}

        def _on_quantum_done(fut: asyncio.Future) -> None:
            try:
                results["quantum"] = fut.result()
            except Exception as exc:
                errors["quantum"] = exc
            quantum_done.set()

        def _on_classical_done(fut: asyncio.Future) -> None:
            try:
                results["classical"] = fut.result()
            except Exception as exc:
                errors["classical"] = exc
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
            if "classical" in pending and (
                "quantum" not in pending or not classical_done.is_set()
            ):
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
                if "quantum" in errors:
                    raise errors["quantum"]
                q = results["quantum"]
                for step in q.get("steps", []):
                    yield {"type": "quantum_step", "data": step}

            if "classical" in pending and classical_done.is_set() and "quantum" not in pending:
                pending.discard("classical")
                if "classical" in errors:
                    raise errors["classical"]
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
