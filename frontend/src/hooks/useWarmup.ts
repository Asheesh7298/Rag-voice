import { useCallback, useEffect, useState } from "react";
import { getHealth } from "../api/client";

const POLL_INTERVAL_MS = 1500;
const MAX_ATTEMPTS = 60;

/**
 * Wakes the backend on page load: polls /health until it answers "ok" (the server scales to zero, so the
 * first visitor waits for it to start) and counts seconds for the loading screen.
 */
export function useWarmup() {
  const [seconds, setSeconds] = useState(0);
  const [ready, setReady] = useState(false);        // backend answered
  const [dismissed, setDismissed] = useState(false); // loading screen hidden (ready, or skipped)

  useEffect(() => {
    if (ready || dismissed) return;
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [ready, dismissed]);

  useEffect(() => {
    const state = { cancelled: false };
    (async () => {
      for (let attempt = 0; attempt < MAX_ATTEMPTS && !state.cancelled; attempt++) {
        try {
          if ((await getHealth()).status === "ok") {
            if (state.cancelled) return;
            setReady(true);
            setTimeout(() => !state.cancelled && setDismissed(true), 650);
            return;
          }
        } catch {
          // the container is still starting; try again
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      }
    })();
    return () => {
      state.cancelled = true;
    };
  }, []);

  return {
    seconds,
    ready,
    dismissed,
    skip: useCallback(() => setDismissed(true), []),
    markReady: useCallback(() => setReady(true), []),
  };
}
