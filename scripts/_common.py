"""Shared helpers for scripts/run_*.py CLIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def print_result(result: dict[str, Any], output: str = "summary") -> None:
    """Emit a solver result dict either as JSON or a human summary."""
    if output == "json":
        print(json.dumps(result, default=_json_default, indent=2))
        return

    meta = result.get("metadata", {})
    steps = result.get("steps", [])
    final = result.get("final_result", {})

    elapsed = meta.get("elapsed")
    elapsed_str = f"{elapsed:.4f}s" if isinstance(elapsed, (int, float)) else "n/a"
    print(f"steps={len(steps)} elapsed={elapsed_str}")
    if meta:
        print("metadata:")
        for k, v in meta.items():
            print(f"  {k}: {_fmt(v)}")
    if final:
        print("final_result:")
        for k, v in final.items():
            print(f"  {k}: {_fmt(v)}")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"[{len(v)} items]"
    if isinstance(v, dict) and len(v) > 8:
        return f"{{{len(v)} keys}}"
    return str(v)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
