import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ModuleSelector from "./components/ModuleSelector.jsx";
import RacePanel from "./components/RacePanel.jsx";
import TimelineScrubber from "./components/TimelineScrubber.jsx";
import Annotations from "./components/Annotations.jsx";
import { generateInitialState } from "./initialState.js";
import "./App.css";

const RACE_STATES = { IDLE: "idle", RUNNING: "running", COMPLETE: "complete" };

// Empty string → same-origin (local docker-compose / nginx proxy).
// Set VITE_BACKEND_URL at build time (Vercel env) to point at Railway.
const API_BASE = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/$/, "");
const WS_BASE = API_BASE
  ? API_BASE.replace(/^http/, "ws")
  : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;

// Default animation duration for the full race (seconds).
// The actual per-step delay = BASE_DURATION / totalSteps.
const BASE_DURATION_S = 6;

const SPEED_OPTIONS = [
  { label: "0.25x", value: 0.25 },
  { label: "0.5x",  value: 0.5 },
  { label: "1x",    value: 1 },
  { label: "2x",    value: 2 },
  { label: "4x",    value: 4 },
];

export default function App() {
  const [modules, setModules] = useState([]);
  const [selectedModule, setSelectedModule] = useState(null);
  const [config, setConfig] = useState({});
  const [raceState, setRaceState] = useState(RACE_STATES.IDLE);

  // Raw steps from backend (arrive all at once when solver finishes)
  const [rawQuantumSteps, setRawQuantumSteps] = useState([]);
  const [rawClassicalSteps, setRawClassicalSteps] = useState([]);

  // Revealed steps (drip-fed to the UI at controlled speed)
  const [revealedQ, setRevealedQ] = useState(0);
  const [revealedC, setRevealedC] = useState(0);

  const [quantumResult, setQuantumResult] = useState(null);
  const [classicalResult, setClassicalResult] = useState(null);
  const [viewIndex, setViewIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeAnnoTab, setActiveAnnoTab] = useState(null);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [raceComplete, setRaceComplete] = useState(false); // backend done, drip may still be running
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const wsRef = useRef(null);
  const playTimerRef = useRef(null);
  const dripTimerRef = useRef(null);

  // Generate initial state whenever module or config changes
  const initialState = useMemo(() => {
    if (!selectedModule) return null;
    return generateInitialState(selectedModule.id, config);
  }, [selectedModule, config]);

  // Visible steps = clamped to revealed count during animation
  const visibleQuantumSteps = useMemo(() => {
    if (raceState === RACE_STATES.IDLE) return initialState?.quantum || [];
    return rawQuantumSteps.slice(0, revealedQ);
  }, [raceState, initialState, rawQuantumSteps, revealedQ]);

  const visibleClassicalSteps = useMemo(() => {
    if (raceState === RACE_STATES.IDLE) return initialState?.classical || [];
    return rawClassicalSteps.slice(0, revealedC);
  }, [raceState, initialState, rawClassicalSteps, revealedC]);

  const displayQuantumResult = raceState === RACE_STATES.IDLE
    ? (initialState?.quantumResult || null) : quantumResult;
  const displayClassicalResult = raceState === RACE_STATES.IDLE
    ? (initialState?.classicalResult || null) : classicalResult;

  // Where the quantum circuit actually executed. The WS path nests this under
  // `final_result`, the REST fallback under `result`; surface either so a
  // silent Aer fallback (bad key, IonQ outage) is visible instead of hidden.
  const execution =
    displayQuantumResult?.final_result?.execution ||
    displayQuantumResult?.result?.execution ||
    null;
  const EXEC_LABELS = {
    aer: "local simulator (Aer)",
    ionq_simulator: "IonQ cloud emulator",
  };
  const execActualLabel = execution
    ? (EXEC_LABELS[execution.actual] || execution.actual)
    : null;

  // --- Drip-feed animation ---
  // When backend data arrives, gradually reveal steps over BASE_DURATION_S / speedMultiplier.
  useEffect(() => {
    if (raceState !== RACE_STATES.RUNNING && !raceComplete) return;

    const totalQ = rawQuantumSteps.length;
    const totalC = rawClassicalSteps.length;
    const totalSteps = Math.max(totalQ, totalC);

    if (totalSteps === 0) return;

    // Already fully revealed?
    if (revealedQ >= totalQ && revealedC >= totalC) {
      if (raceComplete) {
        setRaceState(RACE_STATES.COMPLETE);
      }
      return;
    }

    const delayMs = (BASE_DURATION_S * 1000) / (totalSteps * speedMultiplier);

    dripTimerRef.current = setTimeout(() => {
      setRevealedQ((prev) => Math.min(prev + 1, totalQ));
      setRevealedC((prev) => Math.min(prev + 1, totalC));
      setViewIndex((prev) => {
        const nextVisible = Math.min(Math.max(revealedQ, revealedC) + 1, totalSteps);
        return Math.max(nextVisible - 1, 0);
      });
    }, delayMs);

    return () => clearTimeout(dripTimerRef.current);
  }, [raceState, raceComplete, rawQuantumSteps.length, rawClassicalSteps.length,
      revealedQ, revealedC, speedMultiplier]);

  // --- Fetch modules ---
  useEffect(() => {
    fetch(`${API_BASE}/api/modules`)
      .then((r) => r.json())
      .then((data) => {
        setModules(data);
        if (data.length > 0) {
          setSelectedModule(data[0]);
          setConfig(data[0].default_params || {});
        }
      })
      .catch((err) => console.error("Failed to fetch modules:", err));
  }, []);

  const resetRace = useCallback(() => {
    setRawQuantumSteps([]);
    setRawClassicalSteps([]);
    setRevealedQ(0);
    setRevealedC(0);
    setQuantumResult(null);
    setClassicalResult(null);
    setViewIndex(0);
    setIsPlaying(false);
    setRaceComplete(false);
  }, []);

  const handleSelectModule = useCallback((mod) => {
    setSelectedModule(mod);
    setConfig(mod.default_params || {});
    setRaceState(RACE_STATES.IDLE);
    resetRace();
    setMobileSidebarOpen(false);
  }, [resetRace]);

  const handleConfigChange = useCallback((key, value) => {
    setConfig((prev) =>
      key && typeof key === "object"
        ? { ...prev, ...key }
        : { ...prev, [key]: value }
    );
    if (raceState !== RACE_STATES.RUNNING) {
      setRaceState(RACE_STATES.IDLE);
      resetRace();
    }
  }, [raceState, resetRace]);

  const startRace = useCallback(() => {
    if (!selectedModule) return;

    setRaceState(RACE_STATES.RUNNING);
    resetRace();

    if (wsRef.current) wsRef.current.close();

    const wsUrl = `${WS_BASE}/ws/race/${selectedModule.id}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => ws.send(JSON.stringify(config));

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "quantum_step":
          setRawQuantumSteps((prev) => [...prev, msg.data]);
          break;
        case "classical_step":
          setRawClassicalSteps((prev) => [...prev, msg.data]);
          break;
        case "complete":
          if (msg.data) {
            if (msg.data.quantum) setQuantumResult(msg.data.quantum);
            if (msg.data.classical) setClassicalResult(msg.data.classical);
          }
          setRaceComplete(true); // drip-feed continues until all steps revealed
          break;
        default:
          break;
      }
    };

    ws.onerror = () => {
      fetch(`${API_BASE}/api/race/${selectedModule.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params: config }),
      })
        .then((r) => r.json())
        .then((data) => {
          setRawQuantumSteps(data.quantum?.steps || []);
          setRawClassicalSteps(data.classical?.steps || []);
          setQuantumResult(data.quantum);
          setClassicalResult(data.classical);
          setRaceComplete(true);
        })
        .catch(() => setRaceState(RACE_STATES.IDLE));
    };

    ws.onclose = () => { wsRef.current = null; };
  }, [selectedModule, config, resetRace]);

  // --- Playback (post-race scrubbing) ---
  const totalSteps = Math.max(visibleQuantumSteps.length, visibleClassicalSteps.length);

  useEffect(() => {
    if (isPlaying && raceState === RACE_STATES.COMPLETE) {
      if (viewIndex >= totalSteps - 1) {
        setIsPlaying(false);
        return;
      }
      const delayMs = (BASE_DURATION_S * 1000) / (totalSteps * speedMultiplier);
      playTimerRef.current = setTimeout(() => {
        setViewIndex((i) => Math.min(i + 1, totalSteps - 1));
      }, delayMs);
      return () => clearTimeout(playTimerRef.current);
    }
  }, [isPlaying, viewIndex, raceState, totalSteps, speedMultiplier]);

  const handlePlayPause = () => {
    if (viewIndex >= totalSteps - 1) {
      setViewIndex(0);
      setIsPlaying(true);
    } else {
      setIsPlaying((p) => !p);
    }
  };

  // --- Shared scale: both panels use the same axes ---
  const sharedScale = useMemo(() => {
    if (!selectedModule) return null;
    const allSteps = [...visibleQuantumSteps, ...visibleClassicalSteps];
    if (allSteps.length === 0) return null;

    if (selectedModule.id === "vqe") {
      const energies = allSteps.map((s) => s.energy ?? s.value).filter((e) => e != null);
      const iterations = allSteps.map((s, i) => s.iteration ?? i);
      const gs = displayQuantumResult?.final_result?.exact_ground_state_energy
        ?? displayClassicalResult?.final_result?.exact_ground_state_energy;
      if (energies.length === 0) return null;
      const eMin = Math.min(...energies, ...(gs != null ? [gs] : []));
      const eMax = Math.max(...energies);
      const pad = (eMax - eMin) * 0.15 || 0.1;
      return {
        xMax: Math.max(...iterations, 1),
        yMin: eMin - pad,
        yMax: eMax + pad,
      };
    }

    if (selectedModule.id === "hamiltonian_sim") {
      let maxProb = 0;
      let nStates = 0;
      for (const s of allSteps) {
        const raw = s.state_probabilities || s.state_probs || s.distribution;
        if (!raw) continue;
        const vals = Array.isArray(raw) ? raw : Object.values(raw);
        for (const v of vals) if (v > maxProb) maxProb = v;
        const n = Array.isArray(raw) ? raw.length : Object.keys(raw).length;
        if (n > nStates) nStates = n;
      }
      return {
        yMax: Math.max(maxProb * 1.15, 0.1),
        nStates,
      };
    }

    return null;
  }, [selectedModule, visibleQuantumSteps, visibleClassicalSteps,
      displayQuantumResult, displayClassicalResult]);

  // --- Status bar text (always-present, no layout shift) ---
  const statusBarContent = useMemo(() => {
    if (!selectedModule) return { mode: "empty", text: "" };

    if (raceState === RACE_STATES.IDLE) {
      const step = initialState?.quantum?.[0] || initialState?.classical?.[0];
      return {
        mode: "setup",
        text: step?.description || selectedModule.description,
      };
    }

    if (raceState === RACE_STATES.RUNNING) {
      const revealed = Math.max(revealedQ, revealedC);
      const total = Math.max(rawQuantumSteps.length, rawClassicalSteps.length);
      return {
        mode: "running",
        text: total > 0
          ? `Simulating... step ${revealed} of ${total}`
          : "Submitting quantum job...",
      };
    }

    return { mode: "complete", text: "Race complete." };
  }, [selectedModule, raceState, initialState, revealedQ, revealedC,
      rawQuantumSteps.length, rawClassicalSteps.length]);

  return (
    <div className={`app-shell ${mobileSidebarOpen ? "mobile-sidebar-open" : ""}`}>
      {/* Backdrop, visible on mobile when drawer is open */}
      <div
        className="mobile-backdrop"
        onClick={() => setMobileSidebarOpen(false)}
      />

      {/* Left sidebar */}
      <nav className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">Q</div>
          <div className="brand-text">
            <span className="brand-title">Quantum Advantage Lab</span>
          </div>
        </div>

        <div className="sidebar-section-label">Algorithms</div>
        <ModuleSelector
          modules={modules}
          selected={selectedModule}
          onSelect={handleSelectModule}
        />

        {selectedModule && (
          <div className="sidebar-controls">
            <div className="sidebar-section-label">Parameters</div>
            <ModuleSelector.Controls
              moduleId={selectedModule.id}
              config={config}
              onConfigChange={handleConfigChange}
            />

            {/* Speed control */}
            <div className="speed-control">
              <span className="speed-label">Playback Speed</span>
              <div className="speed-buttons">
                {SPEED_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    className={`speed-btn ${speedMultiplier === opt.value ? "active" : ""}`}
                    onClick={() => setSpeedMultiplier(opt.value)}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              className="start-btn"
              onClick={startRace}
              disabled={raceState === RACE_STATES.RUNNING}
            >
              <span className="btn-icon">{raceState === RACE_STATES.RUNNING ? "\u23F3" : "\u25B6"}</span>
              {raceState === RACE_STATES.RUNNING ? "Running..." : "Start Race"}
            </button>
          </div>
        )}

        <div className="sidebar-footer">
          <span>Open Source (MIT)</span>
        </div>
      </nav>

      {/* Main content */}
      <main className="main-content">
        {/* Fixed-height status bar — always rendered, content transitions */}
        <div className={`status-bar status-${statusBarContent.mode}`}>
          <button
            className="mobile-menu-btn"
            aria-label="Open menu"
            onClick={() => setMobileSidebarOpen((v) => !v)}
          >
            <span />
            <span />
            <span />
          </button>
          {statusBarContent.mode === "setup" && (
            <span className="status-tag">Problem Setup</span>
          )}
          {statusBarContent.mode === "running" && (
            <span className="status-tag running-tag"><span className="pulse-dot" /> Running</span>
          )}
          {statusBarContent.mode === "complete" && (
            <span className="status-tag">Complete</span>
          )}
          <span className="status-text">{statusBarContent.text}</span>
        </div>

        {execution && (
          <div className={`exec-banner ${execution.fell_back ? "exec-banner--warn" : ""}`}>
            {execution.fell_back
              ? `⚠ ${execution.message || `Requested ${execution.requested} but ran on ${execActualLabel}.`}`
              : `Executed on ${execActualLabel}. Jobs appear under the IonQ account that owns the configured API key.`}
          </div>
        )}

        {selectedModule && (
          <>
            {/* Race panels — always visible */}
            <div className="race-area">
              <div className="race-panels">
                <RacePanel
                  type="quantum"
                  moduleId={selectedModule.id}
                  steps={visibleQuantumSteps}
                  result={displayQuantumResult}
                  raceState={raceState}
                  viewIndex={viewIndex}
                  sharedScale={sharedScale}
                />
                <RacePanel
                  type="classical"
                  moduleId={selectedModule.id}
                  steps={visibleClassicalSteps}
                  result={displayClassicalResult}
                  raceState={raceState}
                  viewIndex={viewIndex}
                  sharedScale={sharedScale}
                />
              </div>

              <aside className={`annotations-sidebar ${activeAnnoTab ? "open" : ""}`}>
                <Annotations
                  moduleId={selectedModule.id}
                  quantumSteps={visibleQuantumSteps}
                  classicalSteps={visibleClassicalSteps}
                  quantumResult={displayQuantumResult}
                  classicalResult={displayClassicalResult}
                  viewIndex={viewIndex}
                  raceState={raceState}
                  activeTab={activeAnnoTab}
                  onTabClick={(id) =>
                    setActiveAnnoTab((cur) => (cur === id ? null : id))
                  }
                />
              </aside>
            </div>

            {/* Bottom timeline */}
            {raceState === RACE_STATES.COMPLETE && totalSteps > 1 && (
              <TimelineScrubber
                currentStep={viewIndex}
                totalSteps={totalSteps}
                isPlaying={isPlaying}
                onSeek={setViewIndex}
                onPlayPause={handlePlayPause}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
