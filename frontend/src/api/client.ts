// Typed client for the VoxLore backend (modal_app.py): GET /health, POST /query, POST /voice-query.

/** The backend this app was built and tested against (Modal app voice-rag-v4); used when VITE_API_URL is not set. */
const DEFAULT_API_URL = "https://rawrmeinkayanosaurushun--voice-rag-v4-voicerag-fastapi-app.us-east.modal.run";

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
  const data = await res.json().catch(() => null);
  if (!res.ok || typeof data?.answer !== "string") {
    throw new Error(data?.error ?? `server returned HTTP ${res.status}`);
  }
  return data as QueryResponse;
}

class ServerError extends Error {}
const RETRY_DELAYS_MS = [3000, 6000];

/**
 * POST with retries for temporary failures: network errors and 5xx responses (for example while Modal
 * replaces a preempted GPU container). 4xx responses are real errors and are not retried.
 */
async function post(path: string, makeBody: () => FormData): Promise<QueryResponse> {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(`${API_URL}${path}`, { method: "POST", body: makeBody() });
      if (res.status >= 500) throw new ServerError(`server returned HTTP ${res.status}`);
      return await parse(res);
    } catch (err) {
      const temporary = err instanceof ServerError || err instanceof TypeError; // TypeError: network failure
      if (!temporary || attempt >= RETRY_DELAYS_MS.length) throw err;
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
    }
  }
}

export async function getHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  return res.json();
}

export async function askText(query: string): Promise<QueryResponse> {
  return post("/query", () => {
    const form = new FormData();
    form.append("query", query);
    return form;
  });
}

export async function askVoice(audio: Blob): Promise<QueryResponse> {
  const extension = audio.type.includes("mp4") ? "m4a" : audio.type.includes("ogg") ? "ogg" : "webm";
  return post("/voice-query", () => {
    const form = new FormData();
    form.append("file", audio, `question.${extension}`);
    return form;
  });
}

/** Total server time: the reported total, or the sum of the stages. */
export function totalMs(t: Timings): number {
  return t.total_ms || (t.embed_ms ?? 0) + (t.search_ms ?? 0) + (t.rerank_ms ?? 0) + (t.qa_ms ?? 0);
}
