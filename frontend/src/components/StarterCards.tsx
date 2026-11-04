import { STARTER_QUESTIONS } from "../data";

/** Example questions on the start screen; hidden once the conversation starts. */
export function StarterCards({ onAsk }: { onAsk: (text: string) => void }) {
  return (
    <div className="starter-cards-box">
      <div className="starter-section-label">
        <span>Verified Example Questions</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--purple-primary)" }}>Curated Indic Examples</span>
      </div>

      <div className="starter-cards-grid">
        {STARTER_QUESTIONS.map((q) => (
          <div
            key={q.text}
            className="starter-prompt-card"
            role="button"
            tabIndex={0}
            onClick={() => onAsk(q.text)}
            onKeyDown={(e) => e.key === "Enter" && onAsk(q.text)}
          >
            <div className={`prompt-icon-badge ${q.lang}`}>{q.label}</div>
            <div className="prompt-card-title">{q.text}</div>
            <div className="prompt-card-arrow">↗</div>
          </div>
        ))}
      </div>
    </div>
  );
}
