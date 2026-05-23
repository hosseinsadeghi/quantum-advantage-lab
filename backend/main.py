"""Quantum Advantage Lab -- FastAPI backend with WebSocket race streaming."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.modules import MODULES

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

# CORS -- wide open for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


@app.get("/api/modules/{module_id}/defaults")
async def module_defaults(module_id: str):
    """Return the default parameters for a specific module."""
    mod = _get_module(module_id)
    return mod.info()


@app.post("/api/race/{module_id}")
async def run_race(module_id: str, body: RaceRequest):
    """Run a full race (non-streaming) and return the results."""
    mod = _get_module(module_id)
    try:
        result = await mod.run(body.params)
    except Exception as exc:
        logger.exception("Race execution failed for %s", module_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "module": module_id,
        "quantum": {
            "steps": result.quantum_steps,
            "result": result.quantum_result,
            "time": result.quantum_time,
        },
        "classical": {
            "steps": result.classical_steps,
            "result": result.classical_result,
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
# Static files -- serve the frontend build if it exists
# ---------------------------------------------------------------------------

_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
