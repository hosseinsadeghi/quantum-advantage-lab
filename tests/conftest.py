"""Shared pytest fixtures and path setup for Quantum Advantage Lab tests."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_SEED = 20260417


@pytest.fixture(autouse=True)
def _seeded_rng():
    """Seed Python/NumPy RNGs before every test for determinism."""
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    yield


@pytest.fixture
def small_n_qubits() -> int:
    """Qubit count kept low enough to run unit tests quickly."""
    return 3


@pytest.fixture
def ionq_api_key() -> str | None:
    """IONQ_API_KEY if present; integration tests skip when missing."""
    return os.environ.get("IONQ_API_KEY")
