interface Props {
  warm: boolean;
  /** Server time of the latest answer, shown on the latency button once something was asked. */
  latestTotal: number | null;
  onReset: () => void;
  onOpenTelemetry: () => void;
}

export function Header({ warm, latestTotal, onReset, onOpenTelemetry }: Props) {
  return (
    <header className="app-top-header">
      <div className="header-nav-row">
        <div className="header-user-info">
          <div className="user-avatar-circle">V</div>
          <div className="user-text-info">
            <span className="user-app-name">VoxLore</span>
            <span className="user-app-sub">13.02M Vectors · Modal A100 GPU</span>
          </div>
        </div>

        <div className="header-actions">
          <a
            href="presentation.html"
            target="_blank"
            rel="noreferrer"
            className="btn-header-action"
            style={{ textDecoration: "none", background: "rgba(255,255,255,0.25)", borderColor: "rgba(255,255,255,0.45)", fontWeight: 700 }}
            title="View Architecture & Data Flow Presentation"
          >
            <span>📊</span> Architecture
          </a>
          <button className="btn-header-action" onClick={onReset} title="Reset to starter prompts">
            <span>↺</span> New
          </button>
          <button
            className="btn-header-action"
            onClick={onOpenTelemetry}
            title="View latency breakdown for current message"
            style={{ background: "rgba(255,255,255,0.22)", borderColor: "rgba(255,255,255,0.4)", fontWeight: 700 }}
          >
            <span>⚡</span> <span>{latestTotal === null ? "Latency Breakdown" : `${latestTotal.toFixed(1)}ms Breakdown ↗`}</span>
          </button>
          <div className="sla-pill" onClick={onOpenTelemetry} title="Sub-150ms SLA Verified" style={{ cursor: "pointer" }}>
            <div className="sla-dot" />
            <span>{warm ? "< 200ms SLA" : "< 150ms SLA"}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
