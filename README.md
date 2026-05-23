# Quantum Advantage Lab

A real-time interactive platform that races quantum algorithms against their classical counterparts, visualizing each solver's progress step by step. The application pairs four foundational quantum algorithms with equivalent classical methods and streams the results side by side through animated D3.js visualizations, making quantum speedups tangible and explorable.

The backend executes actual quantum circuits via Qiskit (on the Aer simulator or optionally on IonQ trapped-ion hardware) while simultaneously running classical solvers, then streams intermediate results to the browser over WebSockets. The frontend reveals each algorithm's steps at a controlled pace so users can watch the race unfold, scrub through the timeline afterward, and read contextual explanations of what is happening at each stage.

---

## Table of Contents

- [Race Modules](#race-modules)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Docker Compose (recommended)](#docker-compose-recommended)
  - [Local Development](#local-development)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [IonQ Hardware Support](#ionq-hardware-support)
- [License](#license)
- [Contact](#contact)

---

## Race Modules

The platform ships with four race modules, each comparing a quantum algorithm against its classical counterpart:

### 1. Grover's Search vs Linear Search

Compares Grover's quantum search algorithm (O(sqrt(N)) queries) against brute-force linear search (O(N) queries). The quantum side builds parameterized Grover circuits with oracle and diffusion operators, tracks amplitude evolution via statevector snapshots at every iteration, and runs the final measurement circuit on the chosen backend. The classical side checks items sequentially. The visualization renders a heatmap of probability amplitudes across the search space.

- **Quantum implementation:** `backend/quantum/grovers.py` -- phase-flip oracle, Grover diffusion, optimal iteration count (floor(pi/4 * sqrt(N))), intermediate statevector snapshots.
- **Classical implementation:** `backend/classical/linear_search.py` -- sequential scan with per-step progress tracking.
- **Visualization:** `frontend/src/visualizations/SearchHeatmap.jsx` -- D3 heatmap of probability amplitudes.

### 2. VQE vs Classical Optimization

Compares the Variational Quantum Eigensolver against classical Nelder-Mead optimization for finding molecular ground-state energies. Supports H2 (2 qubits) and LiH (4 qubits) molecules with Hamiltonians derived from the Bravyi-Kitaev transform. The quantum side uses a hardware-efficient ansatz with Ry/Rz rotation layers and all-to-all RXX entangling gates (native to IonQ's Molmer-Sorensen gate), optimized by COBYLA. The classical side optimizes a Givens-rotation trial wavefunction over the full Hamiltonian matrix. Both sides track energy per iteration and report whether chemical accuracy (+/- 1.6 mHa) is achieved.

- **Quantum implementation:** `backend/quantum/vqe.py` -- hardware-efficient ansatz, statevector energy evaluation, COBYLA optimizer, Hamiltonian coefficients from Kandala et al.
- **Classical implementation:** `backend/classical/gradient_opt.py` -- Givens-rotation parameterized wavefunction, Nelder-Mead optimizer, full matrix construction.
- **Visualization:** `frontend/src/visualizations/EnergyLandscape.jsx` -- energy convergence line chart with ground-state reference line and chemical accuracy band.

### 3. Quantum Walk vs Classical Random Walk

Compares a discrete-time coined quantum walk against a Monte Carlo classical random walk on cycle or complete graphs. The quantum walk uses a Hadamard coin (cycle graph) or Grover coin (complete graph) with conditional shift operators, demonstrating ballistic spreading (standard deviation proportional to t) versus classical diffusive spreading (proportional to sqrt(t)).

- **Quantum implementation:** `backend/quantum/quantum_walks.py` -- Hadamard/Grover coin operators, conditional increment/decrement shift, per-step position distributions via partial trace.
- **Classical implementation:** `backend/classical/random_walk.py` -- Monte Carlo simulation over a transition matrix with configurable trial count.
- **Visualization:** `frontend/src/visualizations/GraphTraversal.jsx` -- radial graph layout with node probability encoding.

### 4. Hamiltonian Simulation vs Matrix Exponentiation

Compares quantum Hamiltonian simulation via first-order Trotter-Suzuki decomposition against classical matrix exponentiation (scipy.linalg.expm). Supports transverse-field Ising and Heisenberg spin-chain models. The quantum side tracks fidelity against the exact evolution at every Trotter step, illustrating the accuracy-depth tradeoff. The classical side reports wall time and FLOP estimates, highlighting the exponential O(2^n) scaling in memory and O(8^n) scaling in computation.

- **Quantum implementation:** `backend/quantum/hamiltonian_sim.py` -- CNOT-staircase Pauli rotation decomposition, Trotter circuit builder, per-step fidelity tracking.
- **Classical implementation:** `backend/classical/matrix_exp.py` -- full Hamiltonian matrix construction, scipy expm per time step, FLOP and memory accounting.
- **Visualization:** `frontend/src/visualizations/MatrixDecay.jsx` -- state probability bar chart with fidelity history overlay.

---

## Architecture

```
Browser (React + D3)
  |
  |--- WebSocket /ws/race/{module_id}   (streaming race progress)
  |--- REST       /api/modules           (module listing & defaults)
  |--- REST       /api/race/{module_id}  (one-shot race execution)
  |
Nginx reverse proxy (production) / Vite dev proxy (development)
  |
FastAPI backend (uvicorn)
  |
  +-- RaceModule base class
  |     +-- run()    -- concurrent blocking execution
  |     +-- stream() -- async generator yielding step messages
  |
  +-- Quantum solvers (Qiskit circuits executed on Aer or IonQ)
  +-- Classical solvers (NumPy / SciPy)
```

The WebSocket protocol works as follows:

1. Server accepts the connection.
2. Client sends a JSON configuration message with algorithm parameters.
3. Server streams JSON messages: `quantum_step`, `classical_step`, and finally `complete`.
4. The frontend drip-feeds received steps to the visualization at a configurable playback speed, then allows timeline scrubbing after the race completes.

If WebSocket connectivity fails, the frontend falls back to the REST endpoint for a single batch response.

---

## Project Structure

```
quantum_advantage_lab/
|-- docker-compose.yml          # Orchestrates backend and frontend services
|-- .gitignore
|-- .dockerignore
|
|-- backend/
|   |-- main.py                 # FastAPI app: REST + WebSocket endpoints
|   |-- Dockerfile              # Python 3.12 image with Qiskit and uvicorn
|   |-- requirements.txt        # Python dependencies
|   |
|   |-- modules/                # Race module definitions
|   |   |-- __init__.py         # Module registry (MODULES dict)
|   |   |-- base.py             # RaceModule ABC with concurrent execution
|   |   |-- search_race.py      # Grover's vs Linear Search
|   |   |-- vqe_race.py         # VQE vs Classical Optimization
|   |   |-- walk_race.py        # Quantum Walk vs Random Walk
|   |   |-- simulation_race.py  # Hamiltonian Sim vs Matrix Exponentiation
|   |
|   |-- quantum/                # Quantum algorithm implementations
|   |   |-- provider.py         # Backend selection (Aer / IonQ), transpilation
|   |   |-- grovers.py          # Grover's search with amplitude tracking
|   |   |-- vqe.py              # VQE with hardware-efficient ansatz
|   |   |-- quantum_walks.py    # Discrete-time coined quantum walks
|   |   |-- hamiltonian_sim.py  # Trotter-Suzuki Hamiltonian simulation
|   |
|   |-- classical/              # Classical algorithm implementations
|       |-- linear_search.py    # Sequential brute-force search
|       |-- gradient_opt.py     # Nelder-Mead molecular optimization
|       |-- random_walk.py      # Monte Carlo random walk
|       |-- matrix_exp.py       # scipy.linalg.expm simulation
|
|-- frontend/
    |-- index.html              # Entry point
    |-- package.json            # Node dependencies (React 18, D3 7, Vite 5)
    |-- vite.config.js          # Dev server with API/WS proxy
    |-- Dockerfile              # Multi-stage build: Node build + Nginx
    |-- nginx.conf              # Reverse proxy to backend for /api and /ws
    |
    |-- src/
        |-- main.jsx            # React root
        |-- App.jsx             # Main app: module selection, race orchestration, layout
        |-- App.css             # Full application styling (dark theme)
        |-- initialState.js     # Pre-race visualization data generators
        |
        |-- components/
        |   |-- ModuleSelector.jsx    # Sidebar nav + parameter controls with tooltips
        |   |-- RacePanel.jsx         # Quantum/classical panel with stats and viz
        |   |-- TimelineScrubber.jsx  # Post-race playback timeline
        |   |-- Annotations.jsx       # Contextual algorithm explanations
        |
        |-- visualizations/
            |-- SearchHeatmap.jsx     # Probability heatmap for Grover's
            |-- EnergyLandscape.jsx   # Energy convergence chart for VQE
            |-- GraphTraversal.jsx    # Radial graph for quantum walks
            |-- MatrixDecay.jsx       # State probability bars for Hamiltonian sim
```

---

## Technology Stack

**Backend:**
- Python 3.12, managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`)
- FastAPI with WebSocket support
- Uvicorn ASGI server
- Qiskit >= 1.0 and Qiskit Aer >= 0.14 for quantum circuit construction and simulation
- NumPy and SciPy for classical solvers and numerical computation
- Optional: `qiskit-ionq` via the `[project.optional-dependencies].ionq` extra (`uv sync --extra ionq`)

**Frontend:**
- React 18
- D3.js 7 for all data visualizations
- Vite 5 for development and production builds
- Inter and JetBrains Mono typefaces

**Infrastructure:**
- Docker and Docker Compose for containerized deployment
- Nginx for production reverse proxy (API and WebSocket passthrough)

---

## Getting Started

### Docker Compose (recommended)

```bash
git clone git@github.com:hosseinsadeghi/quantum-advantage-lab.git
cd quantum-advantage-lab
docker compose up --build
```

The frontend will be available at `http://localhost:3000` and the backend API at `http://localhost:8000`.

### Local Development

**Backend:**

Python dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv
once (`curl -LsSf https://astral.sh/uv/install.sh | sh`) then:

```bash
# From the repo root -- the FastAPI app imports `from backend.x ...` and will fail
# if launched from inside `backend/`.
uv sync                           # creates .venv, installs runtime + dev deps
uv sync --extra ionq              # also install qiskit-ionq for hardware access
./scripts/dev_backend.sh          # equivalent to: uv run uvicorn backend.main:app --reload
```

`pyproject.toml` is the source of truth for dependencies; `backend/requirements.txt`
is kept in sync for the Docker image (which uses vanilla `pip install`). Regenerate
it after editing pyproject with: `uv export --no-hashes --no-dev -o backend/requirements.txt`.

**Frontend:**

```bash
./scripts/dev_frontend.sh         # installs deps on first run, then `npm run dev`
```

The Vite dev server starts at `http://localhost:5173` and automatically proxies `/api` and `/ws` requests to the backend at `http://localhost:8000`.

### Testing

```bash
./scripts/test.sh                 # unit + integration tests (excludes slow/benchmarks)
./scripts/test.sh --benchmarks    # include @pytest.mark.slow benchmark tests
IONQ_API_KEY=... ./scripts/test.sh  # include IonQ cloud-simulator integration tests
uv run pytest tests/unit -k hamiltonian -vv   # ad-hoc equivalent
```

Tests live under `tests/` with sub-trees for `unit/`, `integration/`, `benchmarks/`,
`e2e/`, and `fixtures/`. Markers are defined in `pyproject.toml`. See the Phase 1
scaffolding test at `tests/unit/test_scaffold.py` for the minimum smoke pass.

### Direct CLI

Every quantum solver is runnable without a server via `scripts/run_*.py`:

```bash
./scripts/run_hamiltonian.py --n-qubits 3 --n-steps 4 --time 0.5
./scripts/run_grovers.py --n-qubits 3 --target-state 5
./scripts/run_vqe.py --molecule H2 --max-iterations 50
./scripts/run_walk.py --n-qubits 4 --n-steps 8
./scripts/run_race.py --module hamiltonian_sim --param n_qubits=3
./scripts/ionq_smoke_test.py       # noiseless IonQ cloud-sim connectivity check
./scripts/ionq_emulator_test.py    # Bell-state noise comparison: ideal vs forte-1 emulator
./scripts/ionq_run_hamiltonian.py  # Trotter circuit on emulator: exact vs ideal vs noisy TV distances
```

Pass `--output json` for machine-readable output, or `--use-ionq` to route via
the IonQ cloud simulator (falls back to Aer if `IONQ_API_KEY` is unset).

---

## Configuration

Algorithm parameters are adjustable through the sidebar controls in the UI. Each module exposes:

| Module | Key Parameters |
|---|---|
| Grover's Search | `n_qubits` (2--12), `target_state`, `use_simulator` |
| Quantum Walk | `n_qubits` (2--6), `n_steps` (5--50), graph type, `use_simulator` |
| VQE | molecule (H2/LiH), `max_iterations` (10--200), `use_simulator` |
| Hamiltonian Sim | `n_qubits` (2--8), model (Ising/Heisenberg), `time`, `n_steps`, `use_simulator` |

Playback speed can be set to 0.25x, 0.5x, 1x, 2x, or 4x.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/modules` | List all available race modules with metadata and default parameters |
| GET | `/api/modules/{module_id}/defaults` | Get default parameters for a specific module |
| POST | `/api/race/{module_id}` | Run a full race (non-streaming) and return quantum and classical results |
| WS | `/ws/race/{module_id}` | Stream race progress over WebSocket |

The POST body accepts `{"params": {...}}` to override default parameters.

---

## IonQ Hardware Support

The backend includes a provider abstraction (`backend/quantum/provider.py`) that supports execution on IonQ trapped-ion quantum hardware. To enable it:

1. Install the IonQ provider: `uv sync --extra ionq` (or `pip install qiskit-ionq` if not using uv).
2. Get an API key at **https://cloud.ionq.com** → Settings → API Keys, and export it as `IONQ_API_KEY=ionq_...` (or populate `.env`).
3. Smoke-test connectivity: `./scripts/ionq_emulator_test.py` runs a Bell circuit against both the noiseless simulator and the `forte-1` noise-aware emulator (only IonQ QPU still in production). Retired-device profiles: `--noise-model aria-1 | aria-2`.
4. Toggle "Use Simulator" off in the UI to route a race through IonQ. The provider defaults to the free noise-aware emulator; real-QPU submissions are gated behind an explicit `use_qpu` flag (Phase 1.5.4 plumbing).

The provider handles transpilation to IonQ's native gate set (GPi, GPi2, MS) via an intermediate basis of {rx, ry, rz, rxx}. The hardware-efficient VQE ansatz is specifically designed for IonQ's all-to-all qubit connectivity, using RXX entangling gates that map directly to the native Molmer-Sorensen interaction.

If the IonQ provider is unavailable (missing API key or package), the backend falls back to the local Aer simulator automatically.

---

## License

MIT

---

## Contact

**Hossein Sadeghi**
Email: hosseinsadeghiesfahani@gmail.com
GitHub: [github.com/hosseinsadeghi](https://github.com/hosseinsadeghi)
