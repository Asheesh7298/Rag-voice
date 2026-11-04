import { useEffect, useRef, useState } from "react";
import { askText, askVoice, totalMs, type QueryResponse, type Timings } from "./api/client";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { AssistantMessage, PendingMessage, UserMessage } from "./components/Messages";
import { StarterCards } from "./components/StarterCards";
import { TelemetrySheet } from "./components/TelemetrySheet";
import { WarmupOverlay } from "./components/WarmupOverlay";
import { DEFAULT_HINT } from "./data";
import { useRecorder } from "./hooks/useRecorder";
import { useSpeech } from "./hooks/useSpeech";
import { useWarmup } from "./hooks/useWarmup";
import type { Message } from "./types";

const clock = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

function toAssistant(id: string, data: QueryResponse): Message {
  return {
    id,
    role: "assistant",
    answer: data.answer,
    guardrail: data.guardrail_triggered,
    timings: data.timings_ms ?? null,
    sources: data.sources ?? [],
  };
}

export default function App() {
  const warmup = useWarmup();
  const speech = useSpeech();
  const [messages, setMessages] = useState<Message[]>([]);
  const [latestTimings, setLatestTimings] = useState<Timings | null>(null);
  const [telemetry, setTelemetry] = useState<{ open: boolean; timings: Timings | null }>({ open: false, timings: null });
  const [hint, setHint] = useState(DEFAULT_HINT);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const feedRef = useRef<HTMLElement>(null);
  const nextId = useRef(0);
  const newId = () => `m${nextId.current++}`;

  const replaceMessage = (id: string, message: Message) =>
    setMessages((all) => all.map((m) => (m.id === id ? message : m)));

  const ask = async (text: string) => {
    const pendingId = newId();
    setMessages((all) => [
      ...all,
      { id: newId(), role: "user", text, time: clock() },
      { id: pendingId, role: "pending", warm: warmup.ready },
    ]);
    try {
      const data = await askText(text);
      warmup.markReady();
      replaceMessage(pendingId, toAssistant(pendingId, data));
      if (data.timings_ms) setLatestTimings(data.timings_ms);
    } catch (err) {
      replaceMessage(pendingId, {
        id: pendingId, role: "assistant", answer: `Server connection error: ${errorText(err)}`,
        guardrail: "Connection Error", timings: null, sources: [],
      });
    }
  };

  const askByVoice = async (audio: Blob) => {
    setHint("⏳ Transcribing with Sarvam AI...");
    try {
      const data = await askVoice(audio);
      setMessages((all) => [
        ...all,
        { id: newId(), role: "user", text: `🎤 ${data.transcript || "Voice Audio Query"}`, time: clock() },
        toAssistant(newId(), data),
      ]);
      if (data.timings_ms) setLatestTimings(data.timings_ms);
    } catch (err) {
      setMessages((all) => [
        ...all,
        { id: newId(), role: "assistant", answer: `Voice error: ${errorText(err)}`, guardrail: "STT Error", timings: null, sources: [] },
      ]);
    } finally {
      setHint(DEFAULT_HINT);
    }
  };

  const recorder = useRecorder(canvasRef, askByVoice);

  const reset = () => {
    setMessages([]);
    feedRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  };

  // Keep the newest message in view.
  useEffect(() => {
    const timer = setTimeout(() => {
      if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }, 50);
    return () => clearTimeout(timer);
  }, [messages]);

  return (
    <>
      <WarmupOverlay seconds={warmup.seconds} ready={warmup.ready} hidden={warmup.dismissed} onSkip={warmup.skip} />

      <div className="app-screen-wrapper">
        <div className="chat-app-device">
          <Header
            warm={warmup.ready}
            latestTotal={latestTimings ? totalMs(latestTimings) : null}
            onReset={reset}
            onOpenTelemetry={() => setTelemetry({ open: true, timings: latestTimings })}
          />

          <main ref={feedRef} className="chat-scroll-feed">
            {messages.length === 0 && <StarterCards onAsk={ask} />}
            <div style={{ display: "flex", flexDirection: "column", gap: 16, width: "100%" }}>
              {messages.map((m) =>
                m.role === "user" ? (
                  <UserMessage key={m.id} text={m.text} time={m.time} />
                ) : m.role === "pending" ? (
                  <PendingMessage key={m.id} warm={m.warm} />
                ) : (
                  <AssistantMessage
                    key={m.id}
                    message={m}
                    speaking={speech.speakingId === m.id}
                    onToggleSpeech={speech.toggle}
                    onOpenTelemetry={(timings) => setTelemetry({ open: true, timings })}
                  />
                ),
              )}
            </div>
          </main>

          <Composer
            onSend={ask}
            recording={recorder.recording}
            onToggleRecord={recorder.toggle}
            hint={hint}
            canvasRef={canvasRef}
            speaking={speech.speakingId !== null}
            onStopAudio={speech.stopAll}
          />
        </div>
      </div>

      <TelemetrySheet open={telemetry.open} timings={telemetry.timings} onClose={() => setTelemetry((t) => ({ ...t, open: false }))} />
    </>
  );
}
