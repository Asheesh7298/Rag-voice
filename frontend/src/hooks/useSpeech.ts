import { useCallback, useEffect, useState } from "react";

/** Reads answers aloud with the browser's speech synthesis, one answer at a time. */
export function useSpeech() {
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  const stopAll = useCallback(() => {
    window.speechSynthesis?.cancel();
    setSpeakingId(null);
  }, []);

  const toggle = useCallback(
    (id: string, text: string) => {
      const synth = window.speechSynthesis;
      if (!synth) return;
      if (synth.speaking) {
        synth.cancel();
        if (speakingId === id) {   // pressed Stop on the answer that is playing
          setSpeakingId(null);
          return;
        }
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      // cancel() fires end/error on the previous utterance later; only clear if this one is still current
      const finish = () => setSpeakingId((current) => (current === id ? null : current));
      utterance.onend = finish;
      utterance.onerror = finish;
      setSpeakingId(id);
      synth.speak(utterance);
    },
    [speakingId],
  );

  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  return { speakingId, toggle, stopAll };
}
