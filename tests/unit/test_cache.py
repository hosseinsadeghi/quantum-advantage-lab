"""Unit tests for the file-based result cache."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from backend import cache
from backend.modules.base import RaceModule


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QAL_CACHE_DIR", str(tmp_path / "runs"))
    yield


# ---------------------------------------------------------------------------
# Backend identity & policy
# ---------------------------------------------------------------------------


def test_backend_id_defaults_to_aer():
    assert cache.backend_id({}) == "aer"
    assert cache.backend_id({"use_simulator": True}) == "aer"


def test_backend_id_ionq_emulator():
    bid = cache.backend_id({"use_simulator": False, "noise_model": "forte-1"})
    assert bid == "ionq:emulator:forte-1"


def test_backend_id_ionq_qpu():
    bid = cache.backend_id({"use_qpu": True, "qpu_name": "qpu.forte-1"})
    assert bid == "ionq:qpu:qpu.forte-1"


def test_should_cache_policy():
    assert cache.should_cache("ionq:qpu:qpu.forte-1", 2) is True  # QPU always.
    assert cache.should_cache("aer", 3) is False  # small Aer skipped.
    assert cache.should_cache("aer", cache.SMALL_N) is True  # UI max boundary caches.
    assert cache.should_cache("aer", 20) is True  # large Aer cached.
    assert cache.should_cache("ionq:emulator:forte-1", 5) is False
    assert cache.should_cache("ionq:emulator:forte-1", 15) is True


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------


def test_key_stable_across_calls():
    p = {"n_qubits": 4, "time": 0.5, "n_steps": 10, "model": "ising"}
    assert cache.compute_key("m", p) == cache.compute_key("m", p)


def test_key_ignores_control_flags_and_shots():
    base = {"n_qubits": 4, "time": 0.5}
    k1 = cache.compute_key("m", base)
    k2 = cache.compute_key("m", {**base, "shots": 2048, "add_shots": True, "cache_bypass": True})
    assert k1 == k2


def test_key_differs_when_problem_changes():
    p1 = {"n_qubits": 4, "time": 0.5, "n_steps": 10}
    p2 = {"n_qubits": 4, "time": 0.6, "n_steps": 10}
    assert cache.compute_key("m", p1) != cache.compute_key("m", p2)


def test_key_differs_when_backend_changes():
    p1 = {"n_qubits": 4, "use_simulator": True}
    p2 = {"n_qubits": 4, "use_simulator": False, "noise_model": "forte-1"}
    p3 = {"n_qubits": 4, "use_qpu": True, "qpu_name": "qpu.forte-1"}
    keys = {cache.compute_key("m", p) for p in (p1, p2, p3)}
    assert len(keys) == 3


def test_key_differs_between_modules():
    p = {"n_qubits": 4}
    assert cache.compute_key("vqe", p) != cache.compute_key("hamiltonian_sim", p)


def test_hamiltonian_candidate_keys_include_legacy_chain_variants():
    params = {
        "n_qubits": 8,
        "model": "ising",
        "time": 1.0,
        "n_steps": 10,
        "interaction_pattern": "chain",
        "alpha": 3.0,
        "use_qpu": True,
        "qpu_name": "qpu.forte-enterprise-1",
    }
    legacy = {
        "n_qubits": 8,
        "model": "ising",
        "time": 1.0,
        "n_steps": 10,
        "use_qpu": True,
        "qpu_name": "qpu.forte-enterprise-1",
    }
    keys = cache.candidate_keys("hamiltonian_sim", params)
    assert keys[0] == cache.compute_key("hamiltonian_sim", params)
    assert cache.compute_key("hamiltonian_sim", legacy) in keys


def test_key_stable_across_processes():
    """Key must be process-independent — no hash-randomization leakage."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path.insert(0, {str(repo_root)!r}); "
        "from backend import cache; "
        'print(cache.compute_key("m", {"n_qubits": 4, "time": 0.5, "model": "ising"}))'
    )
    env_a = {"PYTHONHASHSEED": "1", "PATH": ""}
    env_b = {"PYTHONHASHSEED": "42", "PATH": ""}
    out_a = subprocess.check_output([sys.executable, "-c", script], env=env_a).strip()
    out_b = subprocess.check_output([sys.executable, "-c", script], env=env_b).strip()
    assert out_a == out_b and out_a  # non-empty and equal.


# ---------------------------------------------------------------------------
# Storage primitives
# ---------------------------------------------------------------------------


def test_put_get_roundtrip():
    key = cache.compute_key("m", {"n_qubits": 4})
    rec = cache.put(key, {"shots": 100, "counts": {"00": 60, "11": 40}, "final_result": {"ok": True}})
    assert rec["run_id"] and rec["valid"] is True and "created_at" in rec
    got = cache.get(key)
    assert len(got) == 1
    assert got[0]["counts"] == {"00": 60, "11": 40}


def test_put_multiple_records_same_key():
    key = cache.compute_key("m", {"n_qubits": 4})
    cache.put(key, {"shots": 100, "counts": {"00": 100}})
    cache.put(key, {"shots": 200, "counts": {"00": 200}})
    records = cache.get(key)
    assert len(records) == 2
    assert sum(r["shots"] for r in records) == 300


def test_invalidate_hides_from_get():
    key = cache.compute_key("m", {"n_qubits": 4})
    r1 = cache.put(key, {"shots": 100, "counts": {"00": 100}})
    r2 = cache.put(key, {"shots": 50, "counts": {"00": 50}})
    assert cache.invalidate(r1["run_id"]) is True
    remaining = cache.get(key)
    assert len(remaining) == 1
    assert remaining[0]["run_id"] == r2["run_id"]


def test_invalidate_unknown_run_id_returns_false():
    assert cache.invalidate("deadbeef") is False


def test_aggregate_sums_shots_and_merges_counts():
    records = [
        {"shots": 100, "counts": {"00": 60, "11": 40}, "valid": True, "created_at": 1.0,
         "final_result": {"gen": 1}, "metadata": {"m": 1}},
        {"shots": 200, "counts": {"00": 120, "11": 80}, "valid": True, "created_at": 2.0,
         "final_result": {"gen": 2}, "metadata": {"m": 2}},
    ]
    agg = cache.aggregate(records)
    assert agg["shots"] == 300
    assert agg["counts"] == {"00": 180, "11": 120}
    assert agg["final_result"] == {"gen": 2}  # newest wins.
    assert agg["metadata"] == {"m": 2}
    assert agg["n_records"] == 2


def test_aggregate_skips_invalid_records():
    records = [
        {"shots": 100, "counts": {"00": 100}, "valid": False, "created_at": 1.0},
        {"shots": 50, "counts": {"00": 50}, "valid": True, "created_at": 2.0},
    ]
    agg = cache.aggregate(records)
    assert agg["shots"] == 50
    assert agg["n_records"] == 1


def test_aggregate_empty():
    assert cache.aggregate([]) == {}


def test_prune_drops_invalid():
    key = cache.compute_key("m", {"n_qubits": 4})
    r1 = cache.put(key, {"shots": 100})
    cache.put(key, {"shots": 200})
    cache.invalidate(r1["run_id"])
    dropped = cache.prune_invalid()
    assert dropped == 1
    records = cache.get(key)
    assert len(records) == 1
    assert records[0]["shots"] == 200


def test_list_keys_empty_and_populated():
    assert cache.list_keys() == []
    cache.put(cache.compute_key("m", {"n_qubits": 4}), {"shots": 100})
    cache.put(cache.compute_key("m", {"n_qubits": 5}), {"shots": 100})
    assert len(cache.list_keys()) == 2


def test_storage_is_json_lines(tmp_path):
    key = cache.compute_key("m", {"n_qubits": 4})
    cache.put(key, {"shots": 10})
    cache.put(key, {"shots": 20})
    path = cache._path_for(key)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must parse.


# ---------------------------------------------------------------------------
# RaceModule hook integration
# ---------------------------------------------------------------------------


class _FakeRace:
    """Minimal double exercising _run_quantum_maybe_cached logic."""

    module_id = "fake"
    default_params = {}
    calls = 0

    def _merged_params(self, overrides=None):
        return dict(overrides or {})

    def run_quantum(self, params):
        type(self).calls += 1
        if params.get("use_qpu"):
            requested = params.get("qpu_name", "qpu.forte-1")
            execution = {
                "requested": requested,
                "actual": requested,
                "fell_back": False,
            }
        elif params.get("use_simulator", True) is False:
            execution = {
                "requested": "ionq_simulator",
                "actual": "ionq_simulator",
                "fell_back": False,
            }
        else:
            execution = {
                "requested": "aer",
                "actual": "aer",
                "fell_back": False,
            }
        return {
            "steps": [{"i": 0}],
            "final_result": {
                "measured_counts": {"00": params.get("shots", 0)},
                "execution": execution,
            },
            "metadata": {"elapsed": 0.01},
        }


def _hook(params):
    from backend.modules.base import RaceModule

    fake = _FakeRace()
    # Monkey-patch the fake so the inherited hook can use it.
    return RaceModule._run_quantum_maybe_cached(fake, params)


def _cache_status(params):
    from backend.modules.base import RaceModule

    fake = _FakeRace()
    return RaceModule.cache_status(fake, params)


def test_hook_skips_cache_for_small_aer():
    _FakeRace.calls = 0
    p = {"n_qubits": 3, "use_simulator": True, "shots": 100}
    _hook(p)
    _hook(p)
    assert _FakeRace.calls == 2  # no caching → always re-runs.
    assert cache.list_keys() == []


def test_cache_status_reports_not_cacheable_for_small_aer():
    status = _cache_status({"n_qubits": 3, "use_simulator": True})
    assert status["cacheable"] is False
    assert status["has_cached_result"] is False


def test_hook_caches_qpu_and_second_call_hits(monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "fake")
    _FakeRace.calls = 0
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    _hook(p)
    result2 = _hook(p)
    assert _FakeRace.calls == 1  # second call was a cache hit.
    assert result2["metadata"]["cache_hit"] is True
    assert result2["metadata"]["cache_shots"] == 100


def test_cache_status_reports_existing_qpu_record(monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "fake")
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    _hook(p)
    status = _cache_status(p)
    assert status["cacheable"] is True
    assert status["has_cached_result"] is True
    assert status["records"] == 1
    assert status["shots"] == 100


def test_cache_status_finds_legacy_hamiltonian_chain_record(monkeypatch):
    monkeypatch.setattr(_FakeRace, "module_id", "hamiltonian_sim")
    monkeypatch.setattr(
        _FakeRace,
        "default_params",
        {
            "interaction_pattern": "chain",
            "alpha": 3.0,
        },
    )
    legacy = {
        "n_qubits": 8,
        "model": "ising",
        "time": 1.0,
        "n_steps": 10,
        "use_qpu": True,
        "qpu_name": "qpu.forte-enterprise-1",
        "shots": 256,
    }
    key = cache.compute_key("hamiltonian_sim", legacy)
    cache.put(
        key,
        {
            "shots": 256,
            "counts": {"0" * 8: 256},
            "steps": [{"i": 0}],
            "final_result": {
                "measured_counts": {"0" * 8: 256},
                "execution": {
                    "requested": "qpu.forte-enterprise-1",
                    "actual": "qpu.forte-enterprise-1",
                    "fell_back": False,
                },
            },
            "metadata": {"elapsed": 0.01},
        },
    )
    status = _cache_status(
        {
            "n_qubits": 8,
            "model": "ising",
            "time": 1.0,
            "n_steps": 10,
            "interaction_pattern": "chain",
            "alpha": 3.0,
            "use_qpu": True,
            "qpu_name": "qpu.forte-enterprise-1",
        }
    )
    assert status["cacheable"] is True
    assert status["has_cached_result"] is True
    assert status["records"] == 1
    assert status["shots"] == 256
    assert status["matched_key"] == key


def test_hook_replays_legacy_hamiltonian_chain_record(monkeypatch):
    monkeypatch.setattr(_FakeRace, "module_id", "hamiltonian_sim")
    monkeypatch.setattr(
        _FakeRace,
        "default_params",
        {
            "interaction_pattern": "chain",
            "alpha": 3.0,
        },
    )
    legacy = {
        "n_qubits": 8,
        "model": "ising",
        "time": 1.0,
        "n_steps": 10,
        "use_qpu": True,
        "qpu_name": "qpu.forte-enterprise-1",
        "shots": 256,
    }
    key = cache.compute_key("hamiltonian_sim", legacy)
    cache.put(
        key,
        {
            "shots": 256,
            "counts": {"0" * 8: 256},
            "steps": [{"i": 0}],
            "final_result": {
                "measured_counts": {"0" * 8: 256},
                "execution": {
                    "requested": "qpu.forte-enterprise-1",
                    "actual": "qpu.forte-enterprise-1",
                    "fell_back": False,
                },
            },
            "metadata": {"elapsed": 0.01},
        },
    )
    _FakeRace.calls = 0
    result = _hook(
        {
            "n_qubits": 8,
            "model": "ising",
            "time": 1.0,
            "n_steps": 10,
            "interaction_pattern": "chain",
            "alpha": 3.0,
            "use_qpu": True,
            "qpu_name": "qpu.forte-enterprise-1",
            "shots": 256,
        }
    )
    assert _FakeRace.calls == 0
    assert result["metadata"]["cache_hit"] is True
    assert result["metadata"]["cache_key"] == key


def test_hook_add_shots_bypasses_lookup_and_appends(monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "fake")
    _FakeRace.calls = 0
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    _hook(p)
    _hook({**p, "add_shots": True, "shots": 200})
    assert _FakeRace.calls == 2
    key = cache.compute_key("fake", p)
    records = cache.get(key)
    assert len(records) == 2
    assert sum(r["shots"] for r in records) == 300


def test_hook_cache_bypass_still_writes(monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "fake")
    _FakeRace.calls = 0
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    _hook({**p, "cache_bypass": True})
    _hook({**p, "cache_bypass": True})
    assert _FakeRace.calls == 2  # bypass skips lookup.
    key = cache.compute_key("fake", p)
    assert len(cache.get(key)) == 2  # but still wrote records.


class _FellBackRace:
    """Double whose run silently fell back to Aer (provider reports it via the
    ``execution`` info every quantum module surfaces under ``final_result``)."""

    module_id = "fakefb"
    calls = 0

    def run_quantum(self, params):
        type(self).calls += 1
        return {
            "steps": [],
            "final_result": {
                "measured_counts": {"00": params.get("shots", 0)},
                "execution": {
                    "requested": "ionq_simulator",
                    "actual": "aer",
                    "fell_back": True,
                },
            },
            "metadata": {"elapsed": 0.01},
        }


def test_hook_skips_cache_on_silent_aer_fallback():
    # Targeted IonQ but the provider fell back to Aer — detected from what
    # actually executed (execution.fell_back), NOT from the key env. Such a
    # result must never be written under the ionq:* key.
    from backend.modules.base import RaceModule

    _FellBackRace.calls = 0
    fake = _FellBackRace()
    p = {"n_qubits": 20, "use_simulator": False, "noise_model": "forte-1", "shots": 100}
    RaceModule._run_quantum_maybe_cached(fake, p)
    RaceModule._run_quantum_maybe_cached(fake, p)
    assert _FellBackRace.calls == 2  # fell back to Aer → never cached
    assert cache.list_keys() == []


class _WrongBackendRace:
    """Double whose execution metadata does not prove the requested QPU ran."""

    module_id = "fakewrong"
    calls = 0

    def run_quantum(self, params):
        type(self).calls += 1
        requested = params.get("qpu_name", "qpu.forte-1")
        return {
            "steps": [],
            "final_result": {
                "measured_counts": {"00": params.get("shots", 0)},
                "execution": {
                    "requested": requested,
                    "actual": "ionq_simulator",
                    "fell_back": False,
                },
            },
            "metadata": {"elapsed": 0.01},
        }


def test_hook_skips_cache_when_execution_does_not_match_qpu():
    from backend.modules.base import RaceModule

    _WrongBackendRace.calls = 0
    fake = _WrongBackendRace()
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    RaceModule._run_quantum_maybe_cached(fake, p)
    RaceModule._run_quantum_maybe_cached(fake, p)
    assert _WrongBackendRace.calls == 2
    assert cache.list_keys() == []


def test_hook_ignores_invalid_qpu_cache_record_on_lookup():
    key = cache.compute_key("fake", {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1"})
    cache.put(
        key,
        {
            "shots": 100,
            "counts": {"00": 100},
            "final_result": {
                "measured_counts": {"00": 100},
                "execution": {
                    "requested": "qpu.forte-1",
                    "actual": "aer",
                    "fell_back": True,
                },
            },
            "metadata": {"elapsed": 0.01},
        },
    )
    _FakeRace.calls = 0
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    result = _hook(p)
    assert _FakeRace.calls == 1
    assert result["metadata"].get("cache_hit") is not True


def test_hook_caches_qpu_without_ionq_key(monkeypatch):
    # The cache must not depend on credentials: a result that genuinely ran on
    # hardware is cached and replayable even with no IonQ key in the env — so
    # the shared .cache/runs/*.jsonl records work for whoever holds them.
    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    monkeypatch.delenv("IONQ_API_KEY_QAL", raising=False)
    _FakeRace.calls = 0
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1", "shots": 100}
    _hook(p)
    result2 = _hook(p)
    assert _FakeRace.calls == 1  # second call served from cache, no key needed
    assert result2["metadata"]["cache_hit"] is True


def test_hook_strips_api_key_metadata_before_caching(monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "fake")
    p = {"n_qubits": 3, "use_qpu": True, "qpu_name": "qpu.forte-1"}

    class _SanitizedRace(_FakeRace):
        @staticmethod
        def run_quantum(params):
            out = _FakeRace().run_quantum(params)
            out["final_result"]["execution"]["message"] = (
                "Submitted to qpu.forte-1 using API key 'default'."
            )
            out["final_result"]["execution"]["key_name"] = "default"
            return out

    fake = type(
        "Fake",
        (),
        {"module_id": "fake", "run_quantum": staticmethod(_SanitizedRace.run_quantum)},
    )()
    result = RaceModule._run_quantum_maybe_cached(fake, p)
    key = cache.compute_key("fake", p)
    records = cache.get(key)

    assert result["final_result"]["execution"]["key_name"] == "default"
    cached_execution = records[0]["final_result"]["execution"]
    assert "key_name" not in cached_execution
    assert "message" not in cached_execution
