"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_PATH = "/models/face_landmarker.task";

type ScanPhase = "setup" | "calibrating" | "scanning";
type Landmark = { x: number; y: number };
type ColorSample = { t: number; g: number };

// Facial-tension "tells" (unvalidated heuristic, kept as a supporting signal — see pulse below for the real one).
const TELL_WEIGHTS: Record<string, number> = {
  noseSneerLeft: 1.4,
  noseSneerRight: 1.4,
  browDownLeft: 1.1,
  browDownRight: 1.1,
  browOuterUpLeft: 0.7,
  browOuterUpRight: 0.7,
  eyeSquintLeft: 0.9,
  eyeSquintRight: 0.9,
  mouthPressLeft: 1.2,
  mouthPressRight: 1.2,
  mouthPucker: 0.8,
  jawLeft: 0.6,
  jawRight: 0.6,
};

const STRESS_ALERTS = [
  "ELEVATED STRESS RESPONSE",
  "PULSE SPIKE DETECTED",
  "AUTONOMIC AROUSAL RISING",
  "FIGHT-OR-FLIGHT ENGAGED",
  "HEART RATE SURGING",
  "TENSION SIGNATURE HIGH",
];

const ALERT_THRESHOLD = 74;
const ALERT_COOLDOWN_MS = 4000;
const BLINK_WINDOW_MS = 12000;
const HISTORY_LENGTH = 220;

const CALIBRATION_SECONDS = 5;
const BUFFER_WINDOW_MS = 15000;
const MIN_SPAN_MS = 4000;
const RESAMPLE_HZ = 20;
const BPM_MIN = 42;
const BPM_MAX = 200;
const BPM_CONFIDENCE_MIN = 0.12;
const BPM_COMPUTE_INTERVAL_MS = 1000;
const PULSE_BASELINE_SPAN_BPM = 18; // +18 BPM over resting maps to a 100% pulse-elevation reading

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

function makeCanvas(width = 0, height = 0) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * Remote photoplethysmography (rPPG): samples the mean green-channel value
 * over a forehead patch. Blood flow subtly changes skin color each heartbeat
 * (Verkruysse et al. 2008) — this is the same underlying signal a pulse
 * oximeter uses, just read optically instead of with an LED/sensor.
 */
function sampleForeheadGreen(
  video: HTMLVideoElement,
  landmarks: Landmark[],
  samplingCanvas: HTMLCanvasElement,
  videoWidth: number,
  videoHeight: number,
): number | null {
  const top = landmarks[10];
  const brow = landmarks[151];
  const leftBrow = landmarks[70];
  const rightBrow = landmarks[300];
  if (!top || !brow || !leftBrow || !rightBrow) return null;

  const cx = (leftBrow.x + rightBrow.x) / 2;
  const cy = brow.y - (brow.y - top.y) * 0.35;
  const halfW = Math.abs(rightBrow.x - leftBrow.x) * 0.24;
  const halfH = Math.abs(brow.y - top.y) * 0.24;

  const sx = clamp((cx - halfW) * videoWidth, 0, videoWidth - 2);
  const sy = clamp((cy - halfH) * videoHeight, 0, videoHeight - 2);
  const sw = clamp(halfW * 2 * videoWidth, 2, videoWidth - sx);
  const sh = clamp(halfH * 2 * videoHeight, 2, videoHeight - sy);

  const ctx = samplingCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, samplingCanvas.width, samplingCanvas.height);
  const { data } = ctx.getImageData(0, 0, samplingCanvas.width, samplingCanvas.height);
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) sum += data[i + 1];
  return sum / (data.length / 4);
}

/**
 * Estimates heart rate from the sampled green-channel trace: resample to a
 * uniform grid, detrend, window, then search the 42-200 BPM band for the
 * dominant frequency. `confidence` is how much of the signal's power sits at
 * that peak vs. spread across the band — low confidence means "noisy/no
 * reliable pulse," not "high stress."
 */
function estimateBpm(samples: ColorSample[], now: number): { bpm: number; confidence: number } | null {
  const recent = samples.filter((s) => now - s.t <= BUFFER_WINDOW_MS);
  if (recent.length < 30) return null;
  const span = recent[recent.length - 1].t - recent[0].t;
  if (span < MIN_SPAN_MS) return null;

  const n = Math.max(8, Math.floor((span / 1000) * RESAMPLE_HZ));
  const t0 = recent[0].t;
  const grid = new Float32Array(n);
  let si = 0;
  for (let i = 0; i < n; i += 1) {
    const t = t0 + (i / RESAMPLE_HZ) * 1000;
    while (si < recent.length - 2 && recent[si + 1].t < t) si += 1;
    const a = recent[si];
    const b = recent[Math.min(si + 1, recent.length - 1)];
    const frac = b.t > a.t ? clamp((t - a.t) / (b.t - a.t), 0, 1) : 0;
    grid[i] = a.g + (b.g - a.g) * frac;
  }

  const smoothWindow = Math.max(1, Math.round(RESAMPLE_HZ * 1.2));
  const prefix = new Float32Array(n + 1);
  for (let i = 0; i < n; i += 1) prefix[i + 1] = prefix[i] + grid[i];
  const detrended = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const lo = Math.max(0, i - smoothWindow);
    const hi = Math.min(n, i + smoothWindow + 1);
    detrended[i] = grid[i] - (prefix[hi] - prefix[lo]) / (hi - lo);
  }
  for (let i = 0; i < n; i += 1) {
    detrended[i] *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / Math.max(1, n - 1)));
  }

  let bestBpm = 0;
  let bestPower = -1;
  let totalPower = 0;
  for (let bpm = BPM_MIN; bpm <= BPM_MAX; bpm += 1) {
    const f = bpm / 60;
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i += 1) {
      const phase = (2 * Math.PI * f * i) / RESAMPLE_HZ;
      re += detrended[i] * Math.cos(phase);
      im -= detrended[i] * Math.sin(phase);
    }
    const power = re * re + im * im;
    totalPower += power;
    if (power > bestPower) {
      bestPower = power;
      bestBpm = bpm;
    }
  }
  return { bpm: bestBpm, confidence: totalPower > 0 ? bestPower / totalPower : 0 };
}

export default function LieDetector() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const waveformRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const samplingCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const phaseRef = useRef<ScanPhase>("setup");
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const alertTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const baselineRef = useRef<Map<string, number>>(new Map());
  const calibrationSamplesRef = useRef<Map<string, number>[]>([]);
  const smoothedScoreRef = useRef(0);
  const lastAlertAtRef = useRef(0);
  const blinkTimestampsRef = useRef<number[]>([]);
  const wasBlinkingRef = useRef(false);
  const pulseWaveHistoryRef = useRef<number[]>(new Array(HISTORY_LENGTH).fill(50));
  const alertRef = useRef(false);

  const colorSamplesRef = useRef<ColorSample[]>([]);
  const lastSampleAtRef = useRef<number | null>(null);
  const slowEmaRef = useRef<number | null>(null);
  const fastEmaRef = useRef<number | null>(null);
  const ampEmaRef = useRef(0.6);
  const bpmRef = useRef<number | null>(null);
  const bpmConfidentRef = useRef(false);
  const baselineBpmRef = useRef<number | null>(null);
  const pulseElevationRef = useRef(0);
  const lastBpmComputeRef = useRef(0);

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [phase, setPhase] = useState<ScanPhase>("setup");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [stressIndex, setStressIndex] = useState(0);
  const [alert, setAlert] = useState(false);
  const [verdict, setVerdict] = useState("");
  const [faceFound, setFaceFound] = useState(false);
  const [bpm, setBpm] = useState<number | null>(null);
  const [pulseConfident, setPulseConfident] = useState(false);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState("");

  const playBuzzer = useCallback(() => {
    const audioContext = audioContextRef.current ?? new AudioContext();
    audioContextRef.current = audioContext;
    if (audioContext.state === "suspended") void audioContext.resume();

    const start = audioContext.currentTime;
    [0, 0.16].forEach((delay) => {
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.type = "sawtooth";
      oscillator.frequency.setValueAtTime(140, start + delay);
      oscillator.frequency.exponentialRampToValueAtTime(70, start + delay + 0.14);
      gain.gain.setValueAtTime(0.0001, start + delay);
      gain.gain.exponentialRampToValueAtTime(0.18, start + delay + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + delay + 0.15);
      oscillator.connect(gain).connect(audioContext.destination);
      oscillator.start(start + delay);
      oscillator.stop(start + delay + 0.16);
    });
  }, []);

  const setScanPhase = useCallback((next: ScanPhase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const startCalibration = useCallback(() => {
    if (countdownTimerRef.current) return;
    calibrationSamplesRef.current = [];
    colorSamplesRef.current = [];
    lastSampleAtRef.current = null;
    slowEmaRef.current = null;
    fastEmaRef.current = null;
    ampEmaRef.current = 0.6;
    bpmRef.current = null;
    bpmConfidentRef.current = false;
    baselineBpmRef.current = null;
    pulseElevationRef.current = 0;
    setBpm(null);
    setPulseConfident(false);
    setScanPhase("calibrating");

    let remaining = CALIBRATION_SECONDS;
    setCountdown(remaining);
    countdownTimerRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining > 0) {
        setCountdown(remaining);
        return;
      }
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
      setCountdown(null);

      const samples = calibrationSamplesRef.current;
      const baseline = new Map<string, number>();
      if (samples.length) {
        for (const category of Object.keys(TELL_WEIGHTS)) {
          const sum = samples.reduce((acc, sample) => acc + (sample.get(category) ?? 0), 0);
          baseline.set(category, sum / samples.length);
        }
      }
      baselineRef.current = baseline;

      const bpmEstimate = estimateBpm(colorSamplesRef.current, performance.now());
      baselineBpmRef.current = bpmEstimate && bpmEstimate.confidence > BPM_CONFIDENCE_MIN ? bpmEstimate.bpm : null;

      smoothedScoreRef.current = 0;
      blinkTimestampsRef.current = [];
      pulseWaveHistoryRef.current = new Array(HISTORY_LENGTH).fill(50);
      lastBpmComputeRef.current = performance.now();
      setScanPhase("scanning");
    }, 1000);
  }, [setScanPhase]);

  const resetScan = useCallback(() => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = null;
    if (alertTimerRef.current) clearTimeout(alertTimerRef.current);
    alertTimerRef.current = null;
    baselineRef.current = new Map();
    smoothedScoreRef.current = 0;
    blinkTimestampsRef.current = [];
    pulseWaveHistoryRef.current = new Array(HISTORY_LENGTH).fill(50);
    colorSamplesRef.current = [];
    lastSampleAtRef.current = null;
    slowEmaRef.current = null;
    fastEmaRef.current = null;
    ampEmaRef.current = 0.6;
    bpmRef.current = null;
    bpmConfidentRef.current = false;
    baselineBpmRef.current = null;
    pulseElevationRef.current = 0;
    alertRef.current = false;
    setStressIndex(0);
    setAlert(false);
    setCountdown(null);
    setBpm(null);
    setPulseConfident(false);
    setScanPhase("setup");
  }, [setScanPhase]);

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setIsModelLoaded(true);
      } catch (modelError) {
        console.error("Unable to load Stress Scan model", modelError);
        setError("Stress Scan could not load its AI model. Check your connection and reload.");
      }
    }
    void loadModel();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!hasStarted || !isModelLoaded) return;
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const waveform = waveformRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !overlay || !waveform || !landmarker) return;
    const activeVideo = video;

    const samplingCanvas = samplingCanvasRef.current ?? makeCanvas(12, 12);
    samplingCanvasRef.current = samplingCanvas;

    let stopped = false;
    let animationFrameId = 0;
    let lastVideoTime = -1;
    let frameCount = 0;
    let fpsStartedAt = performance.now();

    const drawPulseWaveform = (value: number) => {
      const ctx = waveform.getContext("2d");
      if (!ctx) return;
      const width = waveform.width;
      const height = waveform.height;
      const history = pulseWaveHistoryRef.current;
      history.push(value);
      if (history.length > HISTORY_LENGTH) history.shift();

      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      for (let gridY = 0; gridY <= 4; gridY += 1) {
        const y = (gridY / 4) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.strokeStyle = "#39ff9a";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#39ff9a";
      ctx.shadowBlur = 7;
      ctx.beginPath();
      history.forEach((point, index) => {
        const x = (index / (HISTORY_LENGTH - 1)) * width;
        const y = height - (clamp(point, 0, 100) / 100) * height;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    const drawPulseRing = (
      ctx: CanvasRenderingContext2D,
      landmarks: Landmark[],
      width: number,
      height: number,
      now: number,
      bpmValue: number | null,
      stress: number,
    ) => {
      const left = landmarks[234];
      const right = landmarks[454];
      const top = landmarks[10];
      const bottom = landmarks[152];
      if (!left || !right || !top || !bottom) return;

      const cx = ((left.x + right.x) / 2) * width;
      const cy = ((top.y + bottom.y) / 2) * height;
      const rx = ((Math.abs(right.x - left.x) / 2) * width) * 1.18;
      const ry = ((Math.abs(bottom.y - top.y) / 2) * height) * 1.1;

      const beatHz = (bpmValue ?? 70) / 60;
      const beatPhase = ((now / 1000) * beatHz) % 1;
      const thump = Math.exp(-8 * beatPhase) * 0.05;

      const heat = clamp(stress / 100, 0, 1);
      const color = heat > ALERT_THRESHOLD / 100
        ? `rgba(255, ${Math.round(120 - heat * 60)}, ${Math.round(110 - heat * 60)}, 0.85)`
        : `rgba(${Math.round(120 + heat * 135)}, ${Math.round(220 - heat * 60)}, ${Math.round(180 - heat * 80)}, 0.8)`;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 14 + thump * 200;
      ctx.lineWidth = 2 + thump * 40;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx * (1 + thump), ry * (1 + thump), 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const renderFrame = () => {
      if (stopped) return;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const width = video.videoWidth;
        const height = video.videoHeight;
        if (overlay.width !== width || overlay.height !== height) {
          overlay.width = width;
          overlay.height = height;
        }
        const ctx = overlay.getContext("2d");
        const now = performance.now();

        if (ctx && width && height) {
          ctx.clearRect(0, 0, width, height);
          const result = landmarker.detectForVideo(video, now);
          const landmarks = result.faceLandmarks?.[0];
          const blendshapes = result.faceBlendshapes?.[0]?.categories;

          if (landmarks && blendshapes) {
            setFaceFound(true);
            const current = new Map<string, number>();
            for (const category of blendshapes) current.set(category.categoryName, category.score);

            if (phaseRef.current === "calibrating" || phaseRef.current === "scanning") {
              const green = sampleForeheadGreen(video, landmarks, samplingCanvas, width, height);
              if (green != null) {
                colorSamplesRef.current.push({ t: now, g: green });
                const cutoff = now - BUFFER_WINDOW_MS;
                while (colorSamplesRef.current.length && colorSamplesRef.current[0].t < cutoff) {
                  colorSamplesRef.current.shift();
                }

                const lastSampleAt = lastSampleAtRef.current;
                const dt = lastSampleAt != null ? clamp(now - lastSampleAt, 1, 200) : 33;
                lastSampleAtRef.current = now;
                const alphaSlow = 1 - Math.exp(-dt / 3000);
                const alphaFast = 1 - Math.exp(-dt / 250);
                const alphaAmp = 1 - Math.exp(-dt / 6000);

                if (slowEmaRef.current == null || fastEmaRef.current == null) {
                  slowEmaRef.current = green;
                  fastEmaRef.current = green;
                } else {
                  slowEmaRef.current += (green - slowEmaRef.current) * alphaSlow;
                  fastEmaRef.current += (green - fastEmaRef.current) * alphaFast;
                }
                const waveRaw = fastEmaRef.current - slowEmaRef.current;
                ampEmaRef.current += (Math.abs(waveRaw) - ampEmaRef.current) * alphaAmp;
                const normalizedWave = clamp(waveRaw / (ampEmaRef.current * 3.2 + 0.05), -1, 1);
                drawPulseWaveform(normalizedWave * 50 + 50);
              }
            }

            if (phaseRef.current === "calibrating") {
              calibrationSamplesRef.current.push(current);
            } else if (phaseRef.current === "scanning") {
              const baseline = baselineRef.current;
              let tension = 0;
              for (const [category, weight] of Object.entries(TELL_WEIGHTS)) {
                const value = current.get(category) ?? 0;
                const base = baseline.get(category) ?? 0;
                tension += Math.max(0, value - base) * weight;
              }

              const blinkScore = Math.max(current.get("eyeBlinkLeft") ?? 0, current.get("eyeBlinkRight") ?? 0);
              const isBlinking = blinkScore > 0.55;
              if (isBlinking && !wasBlinkingRef.current) blinkTimestampsRef.current.push(now);
              wasBlinkingRef.current = isBlinking;
              blinkTimestampsRef.current = blinkTimestampsRef.current.filter((t) => now - t < BLINK_WINDOW_MS);
              const blinkBonus = Math.max(0, blinkTimestampsRef.current.length - 4) * 3.5;

              if (now - lastBpmComputeRef.current > BPM_COMPUTE_INTERVAL_MS) {
                lastBpmComputeRef.current = now;
                const estimate = estimateBpm(colorSamplesRef.current, now);
                if (estimate && estimate.confidence > BPM_CONFIDENCE_MIN) {
                  bpmRef.current = bpmRef.current == null ? estimate.bpm : bpmRef.current + (estimate.bpm - bpmRef.current) * 0.35;
                  bpmConfidentRef.current = true;
                  if (baselineBpmRef.current != null && bpmRef.current != null) {
                    pulseElevationRef.current = clamp(
                      ((bpmRef.current - baselineBpmRef.current) / PULSE_BASELINE_SPAN_BPM) * 100,
                      0,
                      100,
                    );
                  }
                  setBpm(Math.round(bpmRef.current));
                  setPulseConfident(true);
                } else {
                  bpmConfidentRef.current = false;
                  setPulseConfident(false);
                }
              }

              const facialComponent = clamp(tension * 140 + blinkBonus, 0, 100);
              const pulseTrusted = bpmConfidentRef.current && baselineBpmRef.current != null;
              const pulseWeight = pulseTrusted ? 0.55 : 0;
              const target = clamp(facialComponent * (1 - pulseWeight) + pulseElevationRef.current * pulseWeight, 0, 100);
              smoothedScoreRef.current += (target - smoothedScoreRef.current) * 0.12;

              const displayScore = clamp(smoothedScoreRef.current + (Math.random() - 0.5) * 2, 0, 100);
              setStressIndex(Math.round(displayScore));

              drawPulseRing(ctx, landmarks, width, height, now, bpmRef.current, smoothedScoreRef.current);

              if (
                smoothedScoreRef.current > ALERT_THRESHOLD &&
                now - lastAlertAtRef.current > ALERT_COOLDOWN_MS
              ) {
                lastAlertAtRef.current = now;
                setVerdict(STRESS_ALERTS[Math.floor(Math.random() * STRESS_ALERTS.length)]);
                alertRef.current = true;
                setAlert(true);
                playBuzzer();
                if (alertTimerRef.current) clearTimeout(alertTimerRef.current);
                alertTimerRef.current = setTimeout(() => {
                  alertRef.current = false;
                  setAlert(false);
                }, 2600);
              }
            }
          } else {
            setFaceFound(false);
          }
        }

        frameCount += 1;
        if (now - fpsStartedAt >= 1000) {
          setFps(Math.round((frameCount * 1000) / (now - fpsStartedAt)));
          frameCount = 0;
          fpsStartedAt = now;
        }
      }
      animationFrameId = requestAnimationFrame(renderFrame);
    };

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (stopped) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        activeVideo.srcObject = stream;
        await activeVideo.play();
        renderFrame();
      } catch (cameraError) {
        console.error("Unable to start camera", cameraError);
        setError("Camera access is required for Stress Scan.");
      }
    }
    void startCamera();
    return () => {
      stopped = true;
      cancelAnimationFrame(animationFrameId);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      activeVideo.srcObject = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasStarted, isModelLoaded, playBuzzer]);

  useEffect(() => {
    if (!hasStarted) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      if (event.key.toLowerCase() === "c" && phaseRef.current !== "calibrating") startCalibration();
      else if (event.key.toLowerCase() === "f") void containerRef.current?.requestFullscreen();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [hasStarted, startCalibration]);

  useEffect(() => () => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    if (alertTimerRef.current) clearTimeout(alertTimerRef.current);
    void audioContextRef.current?.close();
  }, []);

  const startExperience = () => {
    audioContextRef.current ??= new AudioContext();
    resetScan();
    setHasStarted(true);
  };
  const stopExperience = () => {
    setHasStarted(false);
    setFaceFound(false);
    setFps(0);
    resetScan();
  };
  const statusText =
    phase === "setup"
      ? "AWAITING CALIBRATION"
      : phase === "calibrating"
        ? "READING RESTING PULSE…"
        : alert
          ? "ALERT · " + verdict
          : !faceFound
            ? "NO FACE IN FRAME"
            : "STRESS SCAN ACTIVE";

  const pulseText = !faceFound && phase === "scanning"
    ? "PULSE —"
    : bpm != null && pulseConfident
      ? `PULSE ${bpm} BPM`
      : phase === "calibrating"
        ? "READING PULSE…"
        : phase === "scanning"
          ? "PULSE SIGNAL WEAK"
          : "PULSE —";

  return (
    <div className="lie-shell">
      <div ref={containerRef} className={`filter-container lie-container lie-container--${phase} ${alert ? "is-alert" : ""}`}>
        {!isModelLoaded && !error && (
          <div className="loading-overlay lie-loading">
            <div className="spinner" />
            <p>Loading facial tracking model…</p>
          </div>
        )}
        {error && (
          <div className="loading-overlay lie-loading">
            <div className="ghost-error-mark">!</div>
            <p>{error}</p>
          </div>
        )}
        {isModelLoaded && !hasStarted && !error && (
          <div className="loading-overlay start-screen lie-start">
            <span className="lie-orb" aria-hidden="true">◎</span>
            <div>
              <span className="mode-card__eyebrow">BIOSIGNAL LAB · STRESS SCAN</span>
              <h2>Read your pulse, not your mind.</h2>
              <p>Calibrates your resting heart rate from subtle skin-color changes on your forehead (webcam rPPG), then tracks pulse elevation and facial tension as a live stress index.</p>
              <small className="lie-disclaimer">Not a lie detector — nothing is, including professional polygraphs. This measures physiological arousal, not truthfulness.</small>
            </div>
            <button className="btn-primary btn-lie" onClick={startExperience}>Start scan</button>
          </div>
        )}
        <div className="camera-wrapper">
          <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
          <canvas ref={overlayRef} className="camera-canvas" />
          {hasStarted && (
            <>
              <div className="lie-hud">
                <div className={`pulse-badge ${pulseConfident ? "is-live" : ""}`}>
                  <span className="pulse-dot" />
                  {pulseText}
                </div>
                <div className={`lie-status lie-status--${phase} ${alert ? "is-alert" : ""}`}>
                  {statusText}
                  <small>{fps} FPS</small>
                </div>
              </div>

              {phase === "setup" && (
                <div className="plate-guide lie-guide">
                  <span className="plate-guide__icon">◎</span>
                  <strong>Ready when you are</strong>
                  <span>Hold still and face the camera for an {CALIBRATION_SECONDS}s baseline read of your resting pulse and expression.</span>
                  <button className="text-action" onClick={startCalibration}>Start calibration</button>
                </div>
              )}
              {phase === "calibrating" && countdown !== null && (
                <div className="plate-guide lie-guide">
                  <span className="countdown-number">{countdown}</span>
                  <strong>Neutral face, stay still</strong>
                  <span>Reading your resting heart rate and expression…</span>
                </div>
              )}

              {phase === "scanning" && (
                <div className="lie-meter" aria-label={`Stress index ${stressIndex}%`}>
                  <span>STRESS INDEX {stressIndex}%</span>
                  <div><i style={{ width: `${stressIndex}%` }} className={stressIndex > ALERT_THRESHOLD ? "is-hot" : ""} /></div>
                </div>
              )}

              <canvas ref={waveformRef} className="lie-waveform" width={480} height={72} />

              <div className="lie-controls">
                <button className="control-button" onClick={startCalibration} disabled={phase === "calibrating"}>Recalibrate</button>
                <button className="control-button control-button--stop" onClick={stopExperience}>Stop</button>
              </div>
            </>
          )}
        </div>
      </div>
      <div className="ghost-steps" aria-label="Stress Scan keyboard controls">
        <span><b>C</b> recalibrate</span><i /><span><b>F</b> fullscreen</span>
      </div>
    </div>
  );
}
