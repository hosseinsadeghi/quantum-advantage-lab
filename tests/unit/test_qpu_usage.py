"""Unit tests for backend.qpu_usage and the provider's run-wrapping."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from backend import qpu_usage
from backend.quantum import provider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setenv("QAL_QPU_USAGE_LOG", str(tmp_path / "qpu_usage.jsonl"))
    yield


@pytest.fixture(autouse=True)
def _no_real_http(monkeypatch):
    """Stop tests from ever hitting the IonQ API."""
    def _stub(job_id):
        return {
            "cost_usd": 1.23,
            "execution_time_seconds": 4.2,
            "predicted_execution_time_seconds": 5.0,
            "predicted_wait_time_seconds": 1.0,
            "ionq_status": "completed",
            "cost_model": "execution_time",
            "submitted_at": "2026-05-15T15:00:00Z",
            "started_at": "2026-05-15T15:00:05Z",
            "completed_at": "2026-05-15T15:00:09Z",
        }
    monkeypatch.setattr(qpu_usage, "_fetch_job_metadata", _stub)
    yield


def _qpu_backend(name="qpu.forte-1"):
    """Fake backend that mimics qiskit_ionq's IonQ QPU backend interface."""
    submitted = {}

    def _run(circuit, shots=None, **kwargs):
        submitted["circuit"] = circuit
        submitted["shots"] = shots
        submitted["kwargs"] = kwargs
        return _fake_job("job-uuid-1")

    backend = SimpleNamespace(name=name, run=_run)
    backend._submitted = submitted  # for assertions
    return backend


def _fake_job(job_id, status="completed"):
    counts = {"00": 800, "11": 200}
    result = SimpleNamespace(get_counts=lambda: counts)
    return SimpleNamespace(
        job_id=lambda: job_id,
        status=lambda: SimpleNamespace(name=status),
        result=lambda: result,
    )


def _bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ---------------------------------------------------------------------------
# QPU detection
# ---------------------------------------------------------------------------


def test_is_qpu_backend_true_for_qpu_prefix():
    assert qpu_usage.is_qpu_backend(SimpleNamespace(name="qpu.forte-1"))
    assert qpu_usage.is_qpu_backend(SimpleNamespace(name=lambda: "qpu.forte-1"))


def test_is_qpu_backend_true_for_qiskit_ionq_prefixed_name():
    # qiskit_ionq returns names like "ionq_qpu.forte-1" — must be detected.
    assert qpu_usage.is_qpu_backend(SimpleNamespace(name="ionq_qpu.forte-1"))
    assert qpu_usage.is_qpu_backend(SimpleNamespace(name=lambda: "ionq_qpu.forte-1"))


def test_is_qpu_backend_false_for_simulator():
    assert not qpu_usage.is_qpu_backend(SimpleNamespace(name="ionq_simulator"))
    assert not qpu_usage.is_qpu_backend(SimpleNamespace(name="aer_simulator"))
    assert not qpu_usage.is_qpu_backend(None)


def test_backend_name_strips_ionq_prefix_in_records():
    # Stored backend name should be canonical (no "ionq_" prefix) so it
    # matches IonQ's REST API naming.
    qpu_usage.record_submission(
        _fake_job("J-prefix"),
        backend=SimpleNamespace(name=lambda: "ionq_qpu.forte-1"),
        circuit=_bell_circuit(),
        shots=10,
    )
    rec = qpu_usage.read_all()[0]
    assert rec["backend"] == "qpu.forte-1"


# ---------------------------------------------------------------------------
# Recording paths
# ---------------------------------------------------------------------------


def test_record_submission_noop_on_non_qpu():
    qpu_usage.record_submission(
        _fake_job("x"),
        backend=SimpleNamespace(name="ionq_simulator"),
        circuit=_bell_circuit(),
        shots=100,
    )
    assert qpu_usage.read_all() == []


def test_record_completion_noop_on_non_qpu():
    qpu_usage.record_completion(
        _fake_job("x"),
        backend=SimpleNamespace(name="aer"),
        result=None,
    )
    assert qpu_usage.read_all() == []


def test_record_submission_writes_jsonl_for_qpu():
    qc = _bell_circuit()
    qpu_usage.record_submission(
        _fake_job("job-A"),
        backend=SimpleNamespace(name="qpu.forte-1"),
        circuit=qc,
        shots=1024,
        run_kwargs={"sampler_seed": 7},
    )
    records = qpu_usage.read_all()
    assert len(records) == 1
    r = records[0]
    assert r["event"] == "submitted"
    assert r["job_id"] == "job-A"
    assert r["backend"] == "qpu.forte-1"
    assert r["shots"] == 1024
    assert r["run_kwargs"] == {"sampler_seed": 7}
    assert r["circuit"]["n_qubits"] == 2
    assert r["circuit"]["two_qubit_gates"] == 1  # cx
    assert r["timestamp_utc"].endswith("+00:00")


def test_record_completion_writes_cost_and_summary():
    job = _fake_job("job-B")
    backend = SimpleNamespace(name="qpu.forte-1")
    qpu_usage.record_submission(job, backend=backend, circuit=_bell_circuit(), shots=100)
    qpu_usage.record_completion(job, backend=backend, result=job.result(), submitted_at=0.0)
    records = qpu_usage.read_all()
    assert len(records) == 2
    completed = records[-1]
    assert completed["event"] == "completed"
    assert completed["cost_usd"] == pytest.approx(1.23)
    assert completed["execution_time_seconds"] == pytest.approx(4.2)
    assert completed["predicted_execution_time_seconds"] == pytest.approx(5.0)
    assert completed["cost_model"] == "execution_time"
    assert completed["started_at_ionq"] == "2026-05-15T15:00:05Z"
    assert completed["completed_at_ionq"] == "2026-05-15T15:00:09Z"
    assert completed["result_summary"]["top_state"] == "00"
    assert completed["result_summary"]["total_shots"] == 1000
    assert completed["wall_time_seconds"] > 0


def test_record_completion_marks_failed_status():
    job = _fake_job("job-C", status="failed")
    backend = SimpleNamespace(name="qpu.forte-1")
    qpu_usage.record_completion(job, backend=backend, result=None)
    rec = qpu_usage.read_all()[0]
    assert rec["event"] == "failed"
    assert rec["status"] == "failed"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_summarize_aggregates_across_jobs():
    backend = SimpleNamespace(name="qpu.forte-1")
    for jid in ("J1", "J2", "J3"):
        j = _fake_job(jid)
        qpu_usage.record_submission(j, backend=backend, circuit=_bell_circuit(), shots=500)
        qpu_usage.record_completion(j, backend=backend, result=j.result(), submitted_at=0.0)
    s = qpu_usage.summarize()
    assert s["total_jobs"] == 3
    assert s["completed_jobs"] == 3
    assert s["total_shots"] == 1500
    assert s["total_cost_usd"] == pytest.approx(3 * 1.23, abs=1e-4)
    assert s["by_backend"]["qpu.forte-1"]["jobs"] == 3


def test_summarize_empty_log():
    s = qpu_usage.summarize()
    assert s["total_jobs"] == 0
    assert s["total_cost_usd"] == 0.0
    assert s["by_backend"] == {}


# ---------------------------------------------------------------------------
# Provider wrapping — the actual chokepoint
# ---------------------------------------------------------------------------


def test_provider_wrapping_records_on_qpu_run():
    qpu = _qpu_backend("qpu.forte-1")
    wrapped = provider._wrap_run_for_usage_logging(qpu)

    job = wrapped.run(_bell_circuit(), shots=2048, noise_model=None)
    # Submission should be written immediately, completion only after .result().
    records = qpu_usage.read_all()
    assert len(records) == 1
    assert records[0]["event"] == "submitted"
    assert records[0]["shots"] == 2048

    _ = job.result()
    records = qpu_usage.read_all()
    assert len(records) == 2
    assert records[1]["event"] == "completed"
    assert records[1]["cost_usd"] == pytest.approx(1.23)


def test_provider_wrapping_is_idempotent():
    qpu = _qpu_backend()
    once = provider._wrap_run_for_usage_logging(qpu)
    twice = provider._wrap_run_for_usage_logging(qpu)
    assert once is twice
    assert once.run is twice.run


def test_provider_wrapping_does_not_double_record_result():
    qpu = _qpu_backend()
    wrapped = provider._wrap_run_for_usage_logging(qpu)
    job = wrapped.run(_bell_circuit(), shots=10)
    job.result()
    job.result()  # second call should not write another completion event
    events = [r["event"] for r in qpu_usage.read_all()]
    assert events == ["submitted", "completed"]


def test_refresh_appends_event_with_latest_cost():
    job = _fake_job("J-refresh")
    backend = SimpleNamespace(name="qpu.forte-1")
    qpu_usage.record_submission(job, backend=backend, circuit=_bell_circuit(), shots=50)
    rec = qpu_usage.refresh("J-refresh")
    assert rec is not None
    assert rec["event"] == "refresh"
    assert rec["cost_usd"] == pytest.approx(1.23)
    assert rec["execution_time_seconds"] == pytest.approx(4.2)
    assert len(qpu_usage.read_all()) == 2


def test_refresh_unknown_job_returns_none():
    assert qpu_usage.refresh("nonexistent") is None


def test_update_cost_rewrites_in_place_no_append():
    job = _fake_job("J-update")
    backend = SimpleNamespace(name="qpu.forte-1")
    qpu_usage.record_submission(job, backend=backend, circuit=_bell_circuit(), shots=42)
    qpu_usage.record_completion(job, backend=backend, result=job.result(), submitted_at=0.0)
    n_before = len(qpu_usage.read_all())
    assert n_before == 2

    diff = qpu_usage.update_cost("J-update")
    assert diff is not None
    assert diff["cost_usd"] == pytest.approx(1.23)

    records = qpu_usage.read_all()
    assert len(records) == n_before  # no new line appended
    for r in records:
        assert r["job_id"] == "J-update"
        assert r["cost_usd"] == pytest.approx(1.23)
        assert r["execution_time_seconds"] == pytest.approx(4.2)
        assert "cost_updated_at_utc" in r


def test_update_cost_unknown_job_returns_none():
    assert qpu_usage.update_cost("nonexistent") is None


def test_refresh_open_jobs_picks_only_open():
    backend = SimpleNamespace(name="qpu.forte-1")
    # J-open has submitted only; J-done has submitted + completed.
    j1 = _fake_job("J-open")
    qpu_usage.record_submission(j1, backend=backend, circuit=_bell_circuit(), shots=10)
    j2 = _fake_job("J-done")
    qpu_usage.record_submission(j2, backend=backend, circuit=_bell_circuit(), shots=10)
    qpu_usage.record_completion(j2, backend=backend, result=j2.result(), submitted_at=0.0)

    refreshed = qpu_usage.refresh_open_jobs()
    assert len(refreshed) == 1
    assert refreshed[0]["job_id"] == "J-open"


def test_log_format_is_one_json_per_line():
    backend = SimpleNamespace(name="qpu.forte-1")
    j = _fake_job("J-line")
    qpu_usage.record_submission(j, backend=backend, circuit=_bell_circuit(), shots=8)
    qpu_usage.record_completion(j, backend=backend, result=j.result(), submitted_at=0.0)
    raw = Path(qpu_usage.log_path()).read_text().splitlines()
    assert len(raw) == 2
    for line in raw:
        assert json.loads(line)  # each line parses as JSON
