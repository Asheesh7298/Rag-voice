import { totalMs, type Timings } from "../api/client";

const SLA_MS = 200;

const STAGES: { key: keyof Timings; label: string; tag: string; bar: string; onlyIfPresent?: boolean }[] = [
  { key: "stt_ms", label: "1. Speech-to-Text", tag: "Sarvam AI · saaras:v3", bar: "stt", onlyIfPresent: true },
  { key: "embed_ms", label: "2. Query Embedding", tag: "Multilingual-E5 · 768d GPU", bar: "embed" },
  { key: "search_ms", label: "3. Dense Vector Search", tag: "GPU Tensor Cores · 13.0M Vectors", bar: "search" },
  { key: "rerank_ms", label: "4. Hybrid Reranking", tag: "BM25 + Morph Trigram", bar: "rerank" },
  { key: "qa_ms", label: "5. Extractive QA Reader", tag: "XLM-RoBERTa · A100", bar: "qa" },
];

interface Props {
  open: boolean;
  timings: Timings | null;
  onClose: () => void;
}

/** Per-stage server latency of one answer, as bars against the 200 ms target. */
export function TelemetrySheet({ open, timings, onClose }: Props) {
  return (
    <div className="telemetry-backdrop" style={{ display: open ? "flex" : "none" }} onClick={onClose}>
      <div className="telemetry-sheet" onClick={(e) => e.stopPropagation()}>
        <div className="telemetry-sheet-head">
          <div className="telemetry-title">
            <span>⚡ PIPELINE TELEMETRY BREAKDOWN</span>
          </div>
          <button className="btn-close-telemetry" onClick={onClose}>
            ✕
          </button>
        </div>
        <div>
          {timings ? (
            <Breakdown timings={timings} />
          ) : (
            <div style={{ padding: "18px 4px", color: "var(--text-muted)", textAlign: "center" }}>
              Ask a question to see its latency breakdown.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Breakdown({ timings }: { timings: Timings }) {
  const total = totalMs(timings);
  const scale = Math.max(160, total, 100);
  const passed = total <= SLA_MS;
  return (
    <>
      <div className="telemetry-sheet-total-bar">
        <span style={{ color: "var(--text-dark)", fontWeight: 700 }}>TOTAL END-TO-END PIPELINE</span>
        <span style={{ color: passed ? "var(--emerald)" : "var(--rose)", fontWeight: 800, fontSize: 13 }}>
          {total.toFixed(1)} ms{"  "}({passed ? "✅ < 200ms SLA PASSED" : "⚠️ SLA EXCEEDED"})
        </span>
      </div>

      <div className="telemetry-stages-flow">
        {STAGES.filter((s) => !s.onlyIfPresent || (timings[s.key] ?? 0) > 0).map((s) => {
          const ms = timings[s.key] ?? 0;
          return (
            <div key={s.key} className="stage-metric-row">
              <div className="stage-metric-labels">
                <div className="stage-name-wrap">
                  <span>{s.label}</span>
                  <span className="stage-tech-tag">{s.tag}</span>
                </div>
                <span className="stage-metric-val">{ms.toFixed(1)} ms</span>
              </div>
              <div className="stage-bar-bg">
                <div className={`stage-bar-fill ${s.bar}`} style={{ width: `${Math.min(100, (ms / scale) * 100)}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
