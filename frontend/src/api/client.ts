// Typed client for the VoxLore backend (modal_app.py): GET /health, POST /query, POST /voice-query.

/** The live backend; used when VITE_API_URL is not set at build time. */
const DEFAULT_API_URL = "https://rawrmeinkayanosaurushun--voice-rag-voicerag-fastapi-app.modal.run";

export const API_URL: string = (
  import.meta.env.VITE_API_URL ??
  (window.location.origin.includes(".modal.run") ? window.location.origin : DEFAULT_API_URL)
).replace(/\/$/, "");

/** Per-stage server timings in milliseconds (stages a guardrail skipped are absent). */
export interface Timings {
  stt_ms?: number;
  embed_ms?: number;
  search_ms?: number;
  rerank_ms?: number;
  qa_ms?: number;
  gen_ms?: number;
  total_ms?: number;
}

export interface Source {
  text: string;
  score: number;
  lang?: string | null;
  lang_name?: string | null;
  strategy?: string | null;
  chunk_id?: string | null;
  query_id?: string | null;
}

export interface QueryResponse {
  query: string;
  transcript: string | null;
  answer: string;
  sources: Source[];
  confidence: number;
  grounded: boolean;
  /** Name of the guardrail that declined the question, or null when it was answered. */
  guardrail_triggered: string | null;
  timings_ms: Timings;
  lang_detected: string | null;
}

async function parse(res: Response): Promise<QueryResponse> {
  const data = await res.json();
  if (!res.ok || typeof data?.answer !== "string") {
    throw new Error(data?.error ?? `server returned HTTP ${res.status}`);
  }
  return data as QueryResponse;
}

export async function getHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  return res.json();
}

export async function askText(query: string): Promise<QueryResponse> {
  const form = new FormData();
  form.append("query", query);
  return parse(await fetch(`${API_URL}/query`, { method: "POST", body: form }));
}

export async function askVoice(audio: Blob): Promise<QueryResponse> {
  const extension = audio.type.includes("mp4") ? "m4a" : audio.type.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("file", audio, `question.${extension}`);
  return parse(await fetch(`${API_URL}/voice-query`, { method: "POST", body: form }));
}

/** Total server time: the reported total, or the sum of the stages. */
export function totalMs(t: Timings): number {
  return t.total_ms || (t.embed_ms ?? 0) + (t.search_ms ?? 0) + (t.rerank_ms ?? 0) + (t.qa_ms ?? 0);
}
