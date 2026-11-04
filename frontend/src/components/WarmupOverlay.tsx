interface Props {
  seconds: number;
  ready: boolean;
  hidden: boolean;
  onSkip: () => void;
}

const STEPS = [
  "Booting Serverless A100 GPU Container",
  "Loading Multilingual-E5 & XLM-RoBERTa Models",
  "Ingesting 13.02M GPU Tensor Vector Index (18.6 GB FP16)",
];

/** Loading screen shown while the backend starts (it scales to zero when idle). */
export function WarmupOverlay({ seconds, ready, hidden, onSkip }: Props) {
  // Which step is active: the steps advance on a timer (3 s, 8 s) until the server answers.
  const stage = ready ? STEPS.length : seconds >= 8 ? 2 : seconds >= 3 ? 1 : 0;
  const progress = ready ? 100 : seconds >= 8 ? Math.min(92, 45 + (seconds - 8) * 6) : seconds >= 3 ? 45 : null;
  const status = ready
    ? "✅ System Ready · Sub-200ms SLA Active!"
    : seconds >= 8
      ? "⚡ Ingesting 13.0M GPU Tensor Vector Index..."
      : seconds >= 3
        ? "🧠 Loading Neural Embedding & QA Models..."
        : "⚡ Initializing Modal A100 GPU...";

  return (
    <div className={`warmup-overlay${hidden ? " hidden-overlay" : ""}`}>
      <div className="warmup-card">
        <div className="warmup-logo-wrap">
          <div className="warmup-logo-avatar">V</div>
          <div className="warmup-logo-title">VoxLore</div>
          <div className="warmup-logo-sub">Multilingual Voice RAG · 13.02M Vectors · Modal A100 GPU</div>
        </div>

        <div className="warmup-progress-box">
          <div className="warmup-status-header">
            <span>{status}</span>
            <span className="warmup-timer">{seconds}s</span>
          </div>
          <div className="warmup-progress-bar-bg">
            <div className="warmup-progress-bar-fill" style={progress === null ? undefined : { width: `${progress}%` }} />
          </div>
        </div>

        <div
          style={{
            background: "rgba(124, 116, 234, 0.1)",
            border: "1px solid rgba(124, 116, 234, 0.3)",
            borderRadius: 12,
            padding: "10px 14px",
            margin: "12px 0 14px",
            fontSize: 11.5,
            color: "var(--text-dark)",
            lineHeight: 1.5,
            textAlign: "center",
          }}
        >
          <span style={{ color: "var(--purple-dark)", fontWeight: 700 }}>⏱️ Please wait, usually under a minute (one-time start):</span>
          <br />
          <span style={{ color: "var(--text-muted)" }}>
            The Modal A100 GPU is loading 13.02M vectors (18.6 GB) into memory. Once loaded, questions are answered
            in <strong>under 100 ms</strong> on the server!
          </span>
        </div>

        <div className="warmup-checklist">
          {STEPS.map((text, i) => (
            <div key={text} className={`warmup-step${i < stage ? " done" : i === stage ? " active" : ""}`}>
              <span className="step-icon">{i < stage ? "✅" : i === stage ? "🔄" : "⏳"}</span>
              <span className="step-text">{text}</span>
            </div>
          ))}
        </div>

        <div className="warmup-footer">
          <button className="btn-skip-warmup" onClick={onSkip}>
            Skip to Chat ↗
          </button>
        </div>
      </div>
    </div>
  );
}
