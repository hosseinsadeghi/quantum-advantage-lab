"""Quantum Advantage Lab -- FastAPI backend with WebSocket race streaming."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load a local .env (IONQ_API_KEY, CORS_ORIGINS, DISABLE_QPU_SUBMISSION, ...)
# for dev runs.
# In Docker/Railway these come from the environment, so a missing .env is fine.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from backend.modules import MODULES
from backend.quantum.hamiltonian_sim import analyze_connectivity
from backend.quantum.provider import QPUUnavailableError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantum_lab")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Quantum Advantage Lab",
    description="Real-time quantum vs classical algorithm racing platform",
    version="0.1.0",
)

# CORS — driven by CORS_ORIGINS env (comma-separated, or "*" for open dev).
_cors_raw = os.environ.get("CORS_ORIGINS", "*").strip()
_cors_origins = ["*"] if _cors_raw == "*" else [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_module(module_id: str):
    cls = MODULES.get(module_id)
    if cls is None:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    return cls()


class RaceRequest(BaseModel):
    """Body for the POST /api/race/{module_id} endpoint."""
    params: dict[str, Any] = {}


class AnalysisRequest(BaseModel):
    """Body for analysis endpoints that accept a params object."""
    params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/modules")
async def list_modules():
    """Return metadata for every available race module."""
    return [cls().info() for cls in MODULES.values()]


@app.get("/api/backends/qpu")
async def qpu_availability():
    """Live availability for each real-hardware IonQ QPU (for the solver picker)."""
    from backend.quantum.provider import get_qpu_availability

    return get_qpu_availability()


@app.get("/api/keys")
async def list_api_keys():
    """List configured IonQ API keys by name (values never exposed) + active one."""
    from backend.quantum.provider import active_key_name, list_keys

    return {"active": active_key_name(), "keys": list_keys()}


class KeySelectRequest(BaseModel):
    name: str


@app.post("/api/keys/select")
async def select_api_key(body: KeySelectRequest):
    """Switch the active IonQ API key by name. Affects subsequent QPU requests."""
    from backend.quantum.provider import active_key_name, list_keys, select_key

    if not select_key(body.name):
        raise HTTPException(
            status_code=404,
            detail=f"No API key named '{body.name}'. Configure IONQ_API_KEY_{body.name} in the environment.",
        )
    return {"active": active_key_name(), "keys": list_keys()}


@app.get("/api/modules/{module_id}/defaults")
async def module_defaults(module_id: str):
    """Return the default parameters for a specific module."""
    mod = _get_module(module_id)
    return mod.info()


@app.post("/api/analysis/hamiltonian/connectivity")
async def connectivity_analysis(body: AnalysisRequest):
    """Return routing/comparison metrics for a Hamiltonian-simulation circuit."""
    try:
        return analyze_connectivity(**body.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cache/{module_id}")
async def cache_status(module_id: str, body: RaceRequest):
    """Return whether the given module params already have a replayable cache entry."""
    mod = _get_module(module_id)
    try:
        return mod.cache_status(body.params)
    except Exception as exc:
        logger.exception("Cache status lookup failed for %s", module_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/race/{module_id}")
async def run_race(module_id: str, body: RaceRequest):
    """Run a full race (non-streaming) and return the results."""
    mod = _get_module(module_id)
    try:
        result = await mod.run(body.params)
    except QPUUnavailableError as exc:
        logger.warning("Race execution failed for %s: %s", module_id, exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Race execution failed for %s", module_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "module": module_id,
        "quantum": {
            "steps": result.quantum_steps,
            "result": result.quantum_result,
            "metadata": result.quantum_metadata,
            "time": result.quantum_time,
        },
        "classical": {
            "steps": result.classical_steps,
            "result": result.classical_result,
            "metadata": result.classical_metadata,
            "time": result.classical_time,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/race/{module_id}")
async def ws_race(websocket: WebSocket, module_id: str):
    """Stream race progress over a WebSocket connection.

    Protocol
    --------
    1. Server accepts the connection.
    2. Client sends a JSON config message, e.g.
       ``{"n_qubits": 4, "use_simulator": true}``
    3. Server streams JSON messages:
       - ``{"type": "quantum_step", "data": {...}}``
       - ``{"type": "classical_step", "data": {...}}``
       - ``{"type": "complete", "data": {...}}``
    4. Server closes the connection after the ``complete`` message.
    """
    cls = MODULES.get(module_id)
    if cls is None:
        await websocket.close(code=4004, reason=f"Module '{module_id}' not found")
        return

    await websocket.accept()
    mod = cls()

    try:
        # Wait for the client to send configuration
        raw = await websocket.receive_text()
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "data": {"message": "Invalid JSON"}})
            await websocket.close(code=4000)
            return

        logger.info("WS race started: module=%s params=%s", module_id, params)

        async for message in mod.stream(params):
            await websocket.send_json(message)

        logger.info("WS race complete: module=%s", module_id)

    except WebSocketDisconnect:
        logger.info("WS client disconnected: module=%s", module_id)
    except Exception as exc:
        logger.exception("WS race error: module=%s", module_id)
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(exc)}})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# QPU hardware-usage log ("the DB")
# ---------------------------------------------------------------------------

@app.get("/api/qpu/usage")
async def qpu_usage_data():
    """Return the recorded real-hardware QPU job log: summary + per-job rows.

    Backs the dashboard at ``/qpu``. Reads the append-only JSONL log written by
    ``backend.qpu_usage`` (Aer simulator / IonQ-emulator runs are never recorded
    there — only jobs that hit a real ``qpu.*`` device).
    """
    from backend import qpu_usage

    return {"summary": qpu_usage.summarize(), "jobs": qpu_usage.list_jobs()}


@app.get("/qpu", response_class=HTMLResponse)
async def qpu_dashboard():
    """Self-contained dashboard for browsing the QPU hardware-usage log."""
    html_path = Path(__file__).resolve().parent / "static" / "qpu_dashboard.html"
    return html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static files -- serve the frontend build if it exists
# ---------------------------------------------------------------------------

_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
