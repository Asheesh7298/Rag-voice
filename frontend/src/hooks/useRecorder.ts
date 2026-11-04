import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

const BACKGROUND = "#F4F6FF";
const BAR_COLORS = ["#7C74EA", "#958EF4"];

/** Prefer formats Sarvam's speech-to-text accepts; the browser picks if none is supported. */
function pickMimeType(): string | undefined {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

/**
 * Records the microphone with MediaRecorder and draws a live waveform on `canvasRef`.
 * When recording stops, the microphone is released and `onAudio` receives the recording.
 */
export function useRecorder(canvasRef: RefObject<HTMLCanvasElement | null>, onAudio: (audio: Blob) => void) {
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const frameRef = useRef<number | null>(null);
  const onAudioRef = useRef(onAudio);
  useEffect(() => {
    onAudioRef.current = onAudio;
  }, [onAudio]);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      ctx.fillStyle = BACKGROUND;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [canvasRef]);

  const stopVisualizer = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    clearCanvas();
  }, [clearCanvas]);

  const startVisualizer = useCallback(
    (stream: MediaStream) => {
      const audioContext = (audioContextRef.current ??= new AudioContext());
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;
      audioContext.createMediaStreamSource(stream).connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);

      const draw = () => {
        frameRef.current = requestAnimationFrame(draw);
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;
        analyser.getByteFrequencyData(data);
        ctx.fillStyle = BACKGROUND;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const barWidth = (canvas.width / data.length) * 1.5;
        data.forEach((value, i) => {
          const height = (value / 255) * (canvas.height - 2);
          ctx.fillStyle = BAR_COLORS[i % 2];
          ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 2, height);
        });
      };
      draw();
    },
    [canvasRef],
  );

  const releaseMicrophone = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
      recorder.onstop = () => {
        stopVisualizer();
        releaseMicrophone();
        onAudioRef.current(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
      };
      streamRef.current = stream;
      recorderRef.current = recorder;
      recorder.start();
      startVisualizer(stream);
      setRecording(true);
    } catch (err) {
      alert("Microphone access required: " + (err instanceof Error ? err.message : String(err)));
    }
  }, [releaseMicrophone, startVisualizer, stopVisualizer]);

  const stop = useCallback(() => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    setRecording(false);
  }, []);

  // Keep the canvas as wide as its container (the old page did this on resize too).
  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      if (canvas?.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 28;
      }
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [canvasRef]);

  useEffect(
    () => () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      releaseMicrophone();
      audioContextRef.current?.close();
    },
    [releaseMicrophone],
  );

  return { recording, toggle: recording ? stop : start };
}
