# Phase 5 results: React + TypeScript frontend

The single-file page (`index.html` + `frontend/index.html`) is replaced by a React 19 + TypeScript + Vite app in
[frontend/](../frontend/). The backend is now API-only (`/health`, `/query`, `/voice-query`).

## Structure

| File | Role |
|---|---|
| `src/api/client.ts` | Typed client for the three endpoints; backend URL from `VITE_API_URL` |
| `src/hooks/useWarmup.ts` | Polls `/health` on page load (the server scales to zero) and drives the loading screen |
| `src/hooks/useRecorder.ts` | Microphone recording and live waveform |
| `src/hooks/useSpeech.ts` | Reads answers aloud, one at a time |
| `src/components/` | WarmupOverlay, Header, StarterCards, Messages (user, pending, assistant, sources), TelemetrySheet, Composer |
| `src/App.tsx` | Chat state and the question flow (text and voice) |
| `src/styles.css` | The original CSS, copied unchanged, plus one phone-width fix at the end |
| `public/presentation.html` | Moved unchanged from `frontend/presentation.html` |

## Fixed along the way

- Answer and source text was inserted as raw HTML; React escapes it.
- The Listen button broke on answers containing an apostrophe (it built `onclick` code from the text).
- The microphone stayed on after recording; it is now released.
- Recordings were labeled `audio/wav` but are WebM/Opus; they are now labeled with their real type.
- The latency button showed made-up sample numbers before any question; it now shows an empty state.
- Outdated text: the page description (A10G GPU, FAISS HNSW, 1.51M vectors), an "A10G" label, and a
  "1–2 minutes" warm-up claim (measured: usually 26–44 s).
- On phones the header buttons ran off the screen (also true of the old page); they now wrap.

## Checks (headless Microsoft Edge, against `voice-rag-v3`)

| Check | Old page | React page |
|---|---|---|
| Loading screen clears when the server is ready | ✅ | ✅ |
| Starter question answered ("Franklin County") | ✅ | ✅ |
| Latency breakdown opens | ✅ | ✅ |
| Sources list opens (3 shown) | ✅ | ✅ |
| Desktop layout (1280 px) | reference | visually identical |
| Phone layout (390 px) | header buttons cut off | all buttons visible |
| Type check, production build, linter (oxlint) | – | ✅ |

Not tested automatically (needs a person): voice recording through a real microphone, and read-aloud.
