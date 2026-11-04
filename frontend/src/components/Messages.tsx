import { useState } from "react";
import { totalMs, type Source, type Timings } from "../api/client";
import type { Message } from "../types";

type Assistant = Extract<Message, { role: "assistant" }>;

export function UserMessage({ text, time }: { text: string; time: string }) {
  return (
    <div className="message-bubble-row user">
      <div className="user-bubble-box">{text}</div>
      <div className="user-time-tag">{time} · You ↗</div>
    </div>
  );
}

export function PendingMessage({ warm }: { warm: boolean }) {
  return (
    <div className="message-bubble-row assistant">
      <div
        className="assistant-bubble-card"
        style={{ color: warm ? "var(--text-muted)" : "var(--purple-dark)", fontWeight: warm ? 400 : 600 }}
      >
        {warm
          ? "⚡ Retrieving across 13,020,220 vectors & extracting grounded answer..."
          : "🔄 One-time initial setup: Waking up Modal A100 GPU & loading 13.02M vectors into VRAM (usually under a minute)... Please wait!"}
      </div>
    </div>
  );
}

interface AssistantProps {
  message: Assistant;
  speaking: boolean;
  onToggleSpeech: (id: string, text: string) => void;
  onOpenTelemetry: (timings: Timings) => void;
}

export function AssistantMessage({ message, speaking, onToggleSpeech, onOpenTelemetry }: AssistantProps) {
  const { id, answer, guardrail, timings, sources } = message;
  const declined = Boolean(guardrail);

  return (
    <div className="message-bubble-row assistant">
      <div className="assistant-bubble-card">
        <div className="assistant-header-row">
          <div className="assistant-avatar-tag">
            <span>VoxLore</span>
            <span className="badge">{declined ? "🛡️ GUARDRAIL" : "🤖 EXTRACTIVE QA"}</span>
          </div>
          {answer && !declined && (
            <button
              className={`btn-read-aloud${speaking ? " speaking" : ""}`}
              onClick={() => onToggleSpeech(id, answer)}
              title="Listen or Stop Audio"
            >
              {speaking ? "⏹ Stop" : "🔊 Listen"}
            </button>
          )}
        </div>

        <div className="assistant-answer-text">{answer}</div>

        {timings && (
          <div className="answer-latency-pill">
            <span>⚡ {totalMs(timings).toFixed(1)} ms</span>
            <button className="btn-view-telemetry" onClick={() => onOpenTelemetry(timings)}>
              Breakdown ↗
            </button>
          </div>
        )}

        {sources.length > 0 && <SourcesAccordion sources={sources} />}
      </div>
    </div>
  );
}

function SourcesAccordion({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="sources-accordion-wrap">
      <span className="sources-accordion-btn" role="button" tabIndex={0} onClick={() => setOpen((o) => !o)}
            onKeyDown={(e) => e.key === "Enter" && setOpen((o) => !o)}>
        📚 {sources.length} Context Sources ▾
      </span>
      <div className="sources-accordion-drawer" style={{ display: open ? "flex" : "none" }}>
        {sources.slice(0, 3).map((s, i) => (
          <div key={s.chunk_id ?? i} className="source-snippet-card">
            <div className="source-snippet-head">
              <span>#{i + 1} · {s.lang_name || s.lang || "indic"}</span>
              <span>Score: {s.score ? s.score.toFixed(3) : ""}</span>
            </div>
            <div>{s.text.slice(0, 160)}...</div>
          </div>
        ))}
      </div>
    </div>
  );
}
