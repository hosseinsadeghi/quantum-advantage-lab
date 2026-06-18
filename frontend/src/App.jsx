import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ModuleSelector from "./components/ModuleSelector.jsx";
import RacePanel from "./components/RacePanel.jsx";
import TimelineScrubber from "./components/TimelineScrubber.jsx";
import Annotations from "./components/Annotations.jsx";
import ConnectivityTab from "./components/ConnectivityTab.jsx";
import AboutPage from "./components/AboutPage.jsx";
import { generateInitialState } from "./initialState.js";
import "./App.css";

const RACE_STATES = { IDLE: "idle", RUNNING: "running", COMPLETE: "complete" };
const CONTENT_TABS = { RACE: "race", CONNECTIVITY: "connectivity" };

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

function defaultExecutionMessage(execution, execActualLabel) {
  if (!execution || !execActualLabel) return "";
  if (execution.actual === "aer") {
    return `Executed on ${execActualLabel}.`;
  }
  if (execution.actual === "ionq_simulator") {
    return `Executed on ${execActualLabel}. Jobs run under the active IonQ API key.`;
  }
  return `Executed on ${execActualLabel}. Jobs appear under the IonQ account that owns the active API key.`;
}

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
  const [activeContentTab, setActiveContentTab] = useState(CONTENT_TABS.RACE);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [raceComplete, setRaceComplete] = useState(false); // backend done, drip may still be running
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [cacheStatus, setCacheStatus] = useState(null);
  const [showAbout, setShowAbout] = useState(true);

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

  // A failed quantum run (e.g. QPU requested but the active key lacks access,
  // and no cached result exists) surfaces its message here — shown loudly so a
  // QPU failure is never silent or mistaken for a simulator run.
  const quantumError =
    displayQuantumResult?.final_result?.error ||
    displayQuantumResult?.result?.error ||
    null;
  const quantumMetadata = displayQuantumResult?.metadata || null;
  const cacheHit = !!quantumMetadata?.cache_hit;
  const showConnectivityPage = selectedModule?.id === "hamiltonian_sim";
  const cacheStatusText = useMemo(() => {
    if (!selectedModule) return "";
    if (cacheHit) {
      return `Cached result in use. Replayed ${quantumMetadata?.cache_shots || 0} total cached shots`
        + (quantumMetadata?.cache_records ? ` across ${quantumMetadata.cache_records} record${quantumMetadata.cache_records === 1 ? "" : "s"}.` : ".");
    }
    if (cacheStatus == null) return "Checking cache status...";
    if (!cacheStatus.cacheable) {
      return "Caching does not apply to the current settings.";
    }
    if (cacheStatus.has_cached_result) {
      return `Cached result available now. ${cacheStatus.shots} cached shots`
        + (cacheStatus.records ? ` across ${cacheStatus.records} record${cacheStatus.records === 1 ? "" : "s"}.` : ".");
    }
    return "No cached result for the current settings.";
  }, [cacheHit, cacheStatus, quantumMetadata, selectedModule]);

  useEffect(() => {
    if (!selectedModule) {
      setCacheStatus(null);
      return undefined;
    }

    let cancelled = false;
    const timeoutId = window.setTimeout(() => {
      fetch(`${API_BASE}/api/cache/${selectedModule.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params: config }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (!cancelled && data) setCacheStatus(data);
        })
        .catch(() => {
          if (!cancelled) {
            setCacheStatus({
              cacheable: false,
              has_cached_result: false,
              lookup_failed: true,
            });
          }
        });
    }, 120);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [config, selectedModule]);

  // --- Drip-feed animation ---
  // When backend data arrives, gradually reveal steps over BASE_DURATION_S / speedMultiplier.
  useEffect(() => {
    if (raceState !== RACE_STATES.RUNNING && !raceComplete) return;

    const totalQ = rawQuantumSteps.length;
    const totalC = rawClassicalSteps.length;
    const totalSteps = Math.max(totalQ, totalC);

    if (totalSteps === 0) {
      if (raceComplete) {
        setRaceState(RACE_STATES.COMPLETE);
      }
      return;
    }

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
    setShowAbout(false);
    setSelectedModule(mod);
    setConfig(mod.default_params || {});
    setRaceState(RACE_STATES.IDLE);
    setActiveContentTab(CONTENT_TABS.RACE);
    setActiveAnnoTab(null);
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

  useEffect(() => {
    if (!showConnectivityPage && activeContentTab !== CONTENT_TABS.RACE) {
      setActiveContentTab(CONTENT_TABS.RACE);
    }
  }, [showConnectivityPage, activeContentTab]);

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
        case "error":
          // Top-level server error — surface it via the quantum result so the
          // error banner shows instead of the race hanging silently.
          setQuantumResult({ final_result: { error: msg.data?.message || "Race failed." } });
          setRaceComplete(true);
          setRaceState(RACE_STATES.COMPLETE);
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
        .then(async (r) => {
          const data = await r.json().catch(() => ({}));
          if (!r.ok) {
            throw new Error(data.detail || data.message || "Race failed.");
          }
          return data;
        })
        .then((data) => {
          setRawQuantumSteps(data.quantum?.steps || []);
          setRawClassicalSteps(data.classical?.steps || []);
          setQuantumResult(data.quantum);
          setClassicalResult(data.classical);
          setRaceComplete(true);
        })
        .catch((err) => {
          setQuantumResult({ final_result: { error: err.message || "Race failed." } });
          setClassicalResult(null);
          setRaceComplete(true);
          setRaceState(RACE_STATES.COMPLETE);
        });
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

    return {
      mode: quantumError ? "error" : "complete",
      text: quantumError
        ? "Quantum run failed."
        : (cacheHit ? "Race complete. Quantum result replayed from cache." : "Race complete."),
    };
  }, [selectedModule, raceState, initialState, revealedQ, revealedC,
      rawQuantumSteps.length, rawClassicalSteps.length, quantumError, cacheHit]);
  const runSummaryItems = useMemo(() => {
    if (!selectedModule) return [];

    const items = [{
      tone: quantumError ? "error" : (
        raceState === RACE_STATES.RUNNING ? "info"
          : raceState === RACE_STATES.COMPLETE ? "success"
            : "neutral"
      ),
      label: quantumError ? "Error" : (
        raceState === RACE_STATES.RUNNING ? "Running"
          : raceState === RACE_STATES.COMPLETE ? "Complete"
            : "Ready"
      ),
      text: quantumError ? `Quantum run failed: ${quantumError}` : statusBarContent.text,
    }];

    if (execution && !quantumError) {
      items.push({
        tone: execution.fell_back ? "warn" : (cacheHit ? "success" : "neutral"),
        label: cacheHit ? "Cache" : "Execution",
        text: execution.fell_back
          ? (execution.message || `Requested ${execution.requested} but ran on ${execActualLabel}.`)
          : (cacheHit
              ? `Replayed cached quantum result from ${execActualLabel}.`
                + (quantumMetadata?.cache_records ? ` Records: ${quantumMetadata.cache_records}.` : "")
                + (quantumMetadata?.cache_shots ? ` Total shots: ${quantumMetadata.cache_shots}.` : "")
                + " No new QPU job was submitted."
              : execution.message
                || defaultExecutionMessage(execution, execActualLabel)),
      });
    }

    items.push({
      tone: cacheStatus?.has_cached_result || cacheHit ? "success" : "subtle",
      label: "Cached Status",
      text: cacheStatus?.lookup_failed
        ? "Could not determine cache status."
        : cacheStatusText,
    });

    return items;
  }, [
    cacheHit,
    cacheStatus,
    cacheStatusText,
    execActualLabel,
    execution,
    quantumError,
    quantumMetadata,
    raceState,
    selectedModule,
    statusBarContent.text,
  ]);

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

        <div className="sidebar-section-label">Project</div>
        <div className="nav-list">
          <button
            className={`nav-item ${showAbout ? "active" : ""}`}
            onClick={() => {
              setShowAbout(true);
              setMobileSidebarOpen(false);
            }}
          >
            <span className="nav-icon">{"ℹ"}</span>
            About this project
          </button>
        </div>

        {selectedModule && !showAbout && (
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

            <div className="run-summary">
              {runSummaryItems.map((item, idx) => (
                <div key={`${item.label}-${idx}`} className={`run-summary-item run-summary-item--${item.tone}`}>
                  <span className="run-summary-label">{item.label}</span>
                  <span className="run-summary-text">{item.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          <span>Open Source (MIT)</span>
        </div>
      </nav>

      {/* Main content */}
      <main className="main-content">
        <div className="mobile-toolbar">
          <button
            className="mobile-menu-btn"
            aria-label="Open menu"
            onClick={() => setMobileSidebarOpen((v) => !v)}
          >
            <span />
            <span />
            <span />
          </button>
          <span className="mobile-toolbar-title">
            {showAbout ? "About" : (selectedModule?.title || "Quantum Advantage Lab")}
          </span>
        </div>

        {showAbout && (
          <section className="content-page">
            <AboutPage
              onLaunch={() => {
                setShowAbout(false);
                if (!selectedModule && modules.length > 0) {
                  handleSelectModule(modules[0]);
                }
              }}
            />
          </section>
        )}

        {!showAbout && selectedModule && (
          <>
            {showConnectivityPage && (
              <div className="content-tabs">
                <button
                  className={`content-tab-btn ${activeContentTab === CONTENT_TABS.RACE ? "active" : ""}`}
                  onClick={() => setActiveContentTab(CONTENT_TABS.RACE)}
                >
                  Race
                </button>
                <button
                  className={`content-tab-btn ${activeContentTab === CONTENT_TABS.CONNECTIVITY ? "active" : ""}`}
                  onClick={() => setActiveContentTab(CONTENT_TABS.CONNECTIVITY)}
                >
                  Connectivity
                </button>
              </div>
            )}

            {activeContentTab === CONTENT_TABS.CONNECTIVITY ? (
              <section className="content-page connectivity-page-shell">
                <ConnectivityTab
                  config={config}
                  apiBase={API_BASE}
                  layout="page"
                />
              </section>
            ) : (
              <>
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
          </>
        )}
      </main>
    </div>
  );
}
