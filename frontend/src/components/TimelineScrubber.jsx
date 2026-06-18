export default function TimelineScrubber({
  currentStep,
  totalSteps,
  isPlaying,
  onSeek,
  onPlayPause,
}) {
  return (
    <div className="timeline-scrubber">
      <button className="timeline-btn" onClick={onPlayPause} title={isPlaying ? "Pause" : "Play"}>
        {isPlaying ? "\u275A\u275A" : "\u25B6"}
      </button>

      <div className="timeline-track">
        <input
          type="range"
          min={0}
          max={Math.max(totalSteps - 1, 0)}
          step={1}
          value={currentStep}
          onChange={(e) => onSeek(parseInt(e.target.value))}
        />
      </div>

      <div className="timeline-info">
        Step {currentStep + 1} / {totalSteps}
      </div>
    </div>
  );
}
