const RACES = [
  {
    name: "Grover's Search vs Linear Search",
    quantum: "O(√N) amplitude amplification",
    classical: "O(N) sequential scan",
    blurb:
      "Watch a phase-flip oracle and the Grover diffusion operator concentrate amplitude on the marked item, while linear search checks one entry at a time.",
  },
  {
    name: "VQE vs Classical Optimization",
    quantum: "Hardware-efficient ansatz + COBYLA",
    classical: "Givens-rotation wavefunction + Nelder-Mead",
    blurb:
      "Both solvers hunt for the molecular ground-state energy of H₂ or LiH, racing toward chemical accuracy (±1.6 mHa) from opposite directions.",
  },
  {
    name: "Quantum Walk vs Random Walk",
    quantum: "Coined discrete-time walk (ballistic)",
    classical: "Monte Carlo random walk (diffusive)",
    blurb:
      "A quantum walker spreads ballistically (σ ∝ t) while its classical cousin only manages diffusive spreading (σ ∝ √t) on the same graph.",
  },
  {
    name: "Hamiltonian Sim vs Matrix Exponentiation",
    quantum: "First-order Trotter–Suzuki circuit",
    classical: "scipy.linalg.expm",
    blurb:
      "Trotterized time evolution is tracked for fidelity at every step, set against exact matrix exponentiation whose cost explodes as O(2ⁿ) memory, O(8ⁿ) compute.",
  },
];

const STEPS = [
  "Pick an algorithm from the sidebar.",
  "Tune its parameters — qubit count, molecule, graph type, evolution time.",
  'Press "Start Race" and watch both solvers stream their progress side by side.',
  "Scrub the timeline afterward to replay any step and read the annotations.",
];

const ROADMAP = [
  {
    tag: "Live today",
    tone: "now",
    items: [
      "Four fully working quantum-vs-classical races with real Qiskit circuits.",
      "Real-time streaming over WebSockets with timeline scrubbing and playback speed.",
      "IonQ cloud emulator execution and native-gate transpilation (GPi / GPi2 / MS).",
      "Connectivity-advantage view for the Hamiltonian module.",
      "Deployed and publicly reachable with repo-backed cache replay for prior hardware runs.",
    ],
  },
  {
    tag: "Next",
    tone: "next",
    items: [
      "Broader curated hardware-cache coverage across more parameter combinations.",
      "More algorithm races (QAOA, phase estimation, error-correction demos).",
      "Shareable race permalinks and side-by-side resource accounting.",
      "Deeper hardware-noise comparisons (ideal vs emulator vs device).",
    ],
  },
];

export default function AboutPage({ onLaunch }) {
  return (
    <div className="about-page">
      <header className="about-hero">
        <div className="about-hero-badge">Quantum Advantage Lab</div>
        <h1 className="about-hero-title">
          See where quantum actually pulls ahead.
        </h1>
        <p className="about-hero-lede">
          An interactive lab that races foundational quantum algorithms against
          their best classical counterparts and streams both, step by step, so
          the speedup is something you watch rather than take on faith.
        </p>
        {onLaunch && (
          <button className="about-launch-btn" onClick={onLaunch}>
            Launch the lab →
          </button>
        )}
      </header>

      <section className="about-section">
        <h2 className="about-section-title">What this is</h2>
        <p className="about-paragraph">
          Quantum speedups are usually described in asymptotic notation that
          hides what is actually happening inside the circuit. This lab runs the
          real thing: the backend builds and executes genuine Qiskit circuits
          (on the Aer simulator or IonQ trapped-ion hardware) while equivalent
          classical solvers run alongside, then streams every intermediate step
          to the browser. You see amplitude concentrate, energy converge, a
          walker spread, or fidelity decay — in real time, side by side with
          the classical method doing the same job.
        </p>
      </section>

      <section className="about-section">
        <h2 className="about-section-title">How to use it</h2>
        <ol className="about-steps">
          {STEPS.map((s, i) => (
            <li key={i} className="about-step">
              <span className="about-step-num">{i + 1}</span>
              <span>{s}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="about-section">
        <h2 className="about-section-title">The four races</h2>
        <div className="about-race-grid">
          {RACES.map((r) => (
            <div key={r.name} className="about-race-card">
              <h3 className="about-race-name">{r.name}</h3>
              <div className="about-race-tags">
                <span className="about-tag about-tag--quantum">{r.quantum}</span>
                <span className="about-tag about-tag--classical">{r.classical}</span>
              </div>
              <p className="about-race-blurb">{r.blurb}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="about-section">
        <h2 className="about-section-title">The quantum component</h2>
        <p className="about-paragraph">
          Every race builds parameterized circuits with Qiskit and tracks their
          internal state at each step — statevector snapshots for Grover, energy
          per iteration for VQE, position distributions for the walk, per-step
          fidelity for Hamiltonian simulation. Circuits transpile to IonQ's
          native trapped-ion gate set, and the VQE ansatz is built around
          all-to-all RXX entanglers that map directly onto IonQ's Mølmer–
          Sørensen interaction. Runs default to the local simulator or the free
          IonQ cloud emulator; real-QPU execution is wired in, and deployments
          can disable fresh hardware submission while still replaying committed
          cached device results.
        </p>
      </section>

      <section className="about-section">
        <h2 className="about-section-title">Where it's going</h2>
        <div className="about-roadmap">
          {ROADMAP.map((col) => (
            <div key={col.tag} className={`about-roadmap-col about-roadmap-col--${col.tone}`}>
              <div className="about-roadmap-tag">{col.tag}</div>
              <ul className="about-roadmap-list">
                {col.items.map((it, i) => (
                  <li key={i}>{it}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <footer className="about-footer">
        <div>
          Built by <strong>Hossein Sadeghi</strong>
        </div>
        <div className="about-footer-links">
          <a href="https://github.com/hosseinsadeghi" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <span className="about-footer-sep">·</span>
          <span>Open Source (MIT)</span>
        </div>
      </footer>
    </div>
  );
}
