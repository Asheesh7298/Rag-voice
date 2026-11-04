import { useState, type RefObject } from "react";
import { QUICK_CHIPS } from "../data";

interface Props {
  onSend: (text: string) => void;
  recording: boolean;
  onToggleRecord: () => void;
  hint: string;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  speaking: boolean;
  onStopAudio: () => void;
}

/** Bottom dock: waveform, text input with mic and send buttons, and quick example chips. */
export function Composer({ onSend, recording, onToggleRecord, hint, canvasRef, speaking, onStopAudio }: Props) {
  const [text, setText] = useState("");

  const send = () => {
    const question = text.trim();
    if (!question) return;
    setText("");
    onSend(question);
  };

  return (
    <div className="app-dock-bottom">
      <div className="waveform-holder">
        <canvas ref={canvasRef} height={28} />
        {!recording && <span className="waveform-hint">{hint}</span>}
        <button
          className="btn-floating-stop"
          onClick={onStopAudio}
          style={{ display: speaking ? "inline-flex" : "none", position: "absolute", right: 8 }}
          title="Stop Voice Speaking"
        >
          <span>⏹</span> Stop Audio
        </button>
      </div>

      <div className="input-dock-capsule">
        <button className={`btn-mic-icon${recording ? " recording" : ""}`} onClick={onToggleRecord} title="Voice Input (Sarvam STT)">
          {recording ? "⏹" : "🎤"}
        </button>
        <input
          type="text"
          className="chat-text-field"
          placeholder="Type your question..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button className="btn-send-round" onClick={send} title="Send">
          ↗
        </button>
      </div>

      <div className="chip-scroll-row">
        {QUICK_CHIPS.map((chip) => (
          <button key={chip.text} className="quick-chip" onClick={() => onSend(chip.text)}>
            <span className="code">{chip.label}</span> {chip.title}
          </button>
        ))}
      </div>
    </div>
  );
}
