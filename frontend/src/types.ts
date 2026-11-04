import type { Source, Timings } from "./api/client";

/** One entry in the chat feed. "pending" is the placeholder shown while a question is being answered. */
export type Message =
  | { id: string; role: "user"; text: string; time: string }
  | { id: string; role: "assistant"; answer: string; guardrail: string | null; timings: Timings | null; sources: Source[] }
  | { id: string; role: "pending"; warm: boolean };
