"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, ImageSegmenter } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const SEGMENTER_MODEL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

type GhostPhase = "setup" | "live" | "vanished";
type Point = { x: number; y: number; z: number };
const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;

function makeCanvas(width = 0, height = 0) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export default function GhostFilter() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const outputRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const segmenterRef = useRef<ImageSegmenter | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const plateRef = useRef<HTMLCanvasElement | null>(null);
  const phaseRef = useRef<GhostPhase>("setup");
  const plateReadyRef = useRef(false);
  const healingRef = useRef(true);
  const handsRef = useRef<Point[][]>([]);
  const pinchLatchedRef = useRef(false);
  const lastPinchAtRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const ghostAmountRef = useRef(0);
  const ghostTargetRef = useRef(0);
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [phase, setPhase] = useState<GhostPhase>("setup");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isPinching, setIsPinching] = useState(false);
  const [handCount, setHandCount] = useState(0);
  const [ghostPercent, setGhostPercent] = useState(0);
  const [fps, setFps] = useState(0);
  const [healing, setHealing] = useState(true);
  const [error, setError] = useState("");

  const setGhostPhase = useCallback((next: GhostPhase) => {
    phaseRef.current = next;
    ghostTargetRef.current = next === "vanished" ? 1 : 0;
    setPhase(next);
  }, []);

  const playWhoosh = useCallback((vanishing: boolean) => {
    const audioContext = audioContextRef.current ?? new AudioContext();
    audioContextRef.current = audioContext;
    if (audioContext.state === "suspended") void audioContext.resume();

    const start = audioContext.currentTime;
    const duration = 0.55;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(vanishing ? 320 : 85, start);
    oscillator.frequency.exponentialRampToValueAtTime(vanishing ? 75 : 360, start + duration);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.12, start + 0.06);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(start);
    oscillator.stop(start + duration);
  }, []);

  const toggleGhost = useCallback(() => {
    if (!plateReadyRef.current) return;
    const vanishing = phaseRef.current !== "vanished";
    setGhostPhase(vanishing ? "vanished" : "live");
    playWhoosh(vanishing);
  }, [playWhoosh, setGhostPhase]);

  const capturePlate = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth) return;
    const plate = plateRef.current ?? makeCanvas();
    plate.width = video.videoWidth;
    plate.height = video.videoHeight;
    plate.getContext("2d")?.drawImage(video, 0, 0, plate.width, plate.height);
    plateRef.current = plate;
    plateReadyRef.current = true;
    setCountdown(null);
    setGhostPhase("live");
  }, [setGhostPhase]);

  const beginPlateCountdown = useCallback(() => {
    if (countdownTimerRef.current || plateReadyRef.current) return;
    let remaining = 3;
    setCountdown(remaining);
    countdownTimerRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining > 0) {
        setCountdown(remaining);
        return;
      }
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
      capturePlate();
    }, 1000);
  }, [capturePlate]);

  const resetPlate = useCallback(() => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = null;
    plateReadyRef.current = false;
    plateRef.current = null;
    ghostAmountRef.current = 0;
    ghostTargetRef.current = 0;
    setGhostPercent(0);
    setCountdown(null);
    setGhostPhase("setup");
  }, [setGhostPhase]);

  const toggleHealing = useCallback(() => {
    healingRef.current = !healingRef.current;
    setHealing(healingRef.current);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadModels() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        const [segmenter, handLandmarker] = await Promise.all([
          ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: SEGMENTER_MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          }),
          HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.48,
            minHandPresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          }),
        ]);
        if (cancelled) {
          segmenter.close();
          handLandmarker.close();
          return;
        }
        segmenterRef.current = segmenter;
        handLandmarkerRef.current = handLandmarker;
        setIsModelLoaded(true);
      } catch (modelError) {
        console.error("Unable to load Ghost Mode models", modelError);
        setError("Ghost Mode could not load its AI models. Check your connection and reload.");
      }
    }
    void loadModels();
    return () => {
      cancelled = true;
      segmenterRef.current?.close();
      handLandmarkerRef.current?.close();
      segmenterRef.current = null;
      handLandmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!hasStarted || !isModelLoaded) return;
    const video = videoRef.current;
    const output = outputRef.current;
    const segmenter = segmenterRef.current;
    const handLandmarker = handLandmarkerRef.current;
    if (!video || !output || !segmenter || !handLandmarker) return;
    const activeVideo = video;

    const maskCanvas = makeCanvas();
    const personMaskCanvas = makeCanvas();
    const ghostLayerCanvas = makeCanvas();
    const safeFrameCanvas = makeCanvas();
    let stopped = false;
    let animationFrameId = 0;
    let lastHandCheck = 0;
    let lastHandSeenAt = 0;
    let frameCount = 0;
    let fpsStartedAt = performance.now();
    let lastPercent = -1;

    const resizeBuffers = (width: number, height: number) => {
      for (const canvas of [output, personMaskCanvas, ghostLayerCanvas, safeFrameCanvas]) {
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
      }
    };

    const drawHandHud = (ctx: CanvasRenderingContext2D, trackedHands: Point[][], width: number, height: number) => {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 190, 132, 0.88)";
      ctx.fillStyle = "#ffd479";
      ctx.lineWidth = Math.max(1.6, width / 700);
      ctx.shadowColor = "rgba(255, 145, 90, 0.75)";
      ctx.shadowBlur = 8;
      for (const hand of trackedHands) {
        for (const connection of HAND_CONNECTIONS) {
          const start = hand[connection.start];
          const end = hand[connection.end];
          if (!start || !end) continue;
          ctx.beginPath();
          ctx.moveTo(start.x * width, start.y * height);
          ctx.lineTo(end.x * width, end.y * height);
          ctx.stroke();
        }
        hand.forEach((point, index) => {
          ctx.beginPath();
          ctx.arc(point.x * width, point.y * height, index === 4 || index === 8 ? 4.5 : 2.2, 0, Math.PI * 2);
          ctx.fill();
        });
      }
      if (trackedHands.length === 2) {
        const centers = trackedHands.map((hand) => {
          const ids = [0, 5, 9, 13, 17];
          const sum = ids.reduce((acc, id) => ({ x: acc.x + hand[id].x, y: acc.y + hand[id].y }), { x: 0, y: 0 });
          return { x: (sum.x / ids.length) * width, y: (sum.y / ids.length) * height };
        });
        const left = Math.min(centers[0].x, centers[1].x);
        const right = Math.max(centers[0].x, centers[1].x);
        const middleY = (centers[0].y + centers[1].y) / 2;
        const portalWidth = Math.max(180, right - left - 30);
        const portalHeight = Math.max(120, portalWidth * 0.58);
        ctx.strokeStyle = "rgba(255, 212, 121, 0.72)";
        ctx.lineWidth = 1.5;
        ctx.strokeRect((left + right - portalWidth) / 2, middleY - portalHeight / 2, portalWidth, portalHeight);
      }
      ctx.restore();
    };

    const renderFrame = () => {
      if (stopped) return;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const width = video.videoWidth;
        const height = video.videoHeight;
        resizeBuffers(width, height);
        const now = performance.now();
        const ctx = output.getContext("2d");
        const personMaskCtx = personMaskCanvas.getContext("2d");
        const ghostLayerCtx = ghostLayerCanvas.getContext("2d");
        const safeFrameCtx = safeFrameCanvas.getContext("2d");

        if (ctx && personMaskCtx && ghostLayerCtx && safeFrameCtx) {
          const segmentation = segmenter.segmentForVideo(video, now);
          const confidenceMask = segmentation.confidenceMasks?.[0];
          if (confidenceMask) {
            const values = confidenceMask.getAsFloat32Array();
            if (maskCanvas.width !== confidenceMask.width || maskCanvas.height !== confidenceMask.height) {
              maskCanvas.width = confidenceMask.width;
              maskCanvas.height = confidenceMask.height;
            }
            const maskCtx = maskCanvas.getContext("2d");
            if (maskCtx) {
              const image = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
              for (let i = 0; i < values.length; i += 1) {
                const confidence = Math.max(0, Math.min(1, (values[i] - 0.12) * 1.7));
                const offset = i * 4;
                image.data[offset] = 255;
                image.data[offset + 1] = 255;
                image.data[offset + 2] = 255;
                image.data[offset + 3] = confidence * 255;
              }
              maskCtx.putImageData(image, 0, 0);
              personMaskCtx.clearRect(0, 0, width, height);
              personMaskCtx.filter = "blur(5px)";
              personMaskCtx.drawImage(maskCanvas, 0, 0, width, height);
              personMaskCtx.filter = "none";

              const plate = plateRef.current;
              if (plateReadyRef.current && plate) {
                if (healingRef.current) {
                  safeFrameCtx.clearRect(0, 0, width, height);
                  safeFrameCtx.drawImage(video, 0, 0, width, height);
                  safeFrameCtx.globalCompositeOperation = "destination-out";
                  safeFrameCtx.drawImage(personMaskCanvas, 0, 0, width, height);
                  safeFrameCtx.globalCompositeOperation = "source-over";
                  const plateCtx = plate.getContext("2d");
                  if (plateCtx) {
                    plateCtx.globalAlpha = 0.018;
                    plateCtx.drawImage(safeFrameCanvas, 0, 0, width, height);
                    plateCtx.globalAlpha = 1;
                  }
                }
                ghostAmountRef.current += (ghostTargetRef.current - ghostAmountRef.current) * 0.09;
                if (Math.abs(ghostTargetRef.current - ghostAmountRef.current) < 0.004) ghostAmountRef.current = ghostTargetRef.current;
                ctx.clearRect(0, 0, width, height);
                ctx.drawImage(video, 0, 0, width, height);
                if (ghostAmountRef.current > 0.004) {
                  ghostLayerCtx.clearRect(0, 0, width, height);
                  ghostLayerCtx.drawImage(plate, 0, 0, width, height);
                  ghostLayerCtx.globalCompositeOperation = "destination-in";
                  ghostLayerCtx.drawImage(personMaskCanvas, 0, 0, width, height);
                  ghostLayerCtx.globalCompositeOperation = "source-over";
                  ctx.globalAlpha = ghostAmountRef.current;
                  ctx.drawImage(ghostLayerCanvas, 0, 0, width, height);
                  ctx.globalAlpha = 1;
                }
              } else {
                ctx.clearRect(0, 0, width, height);
                ctx.drawImage(video, 0, 0, width, height);
              }
            }
          } else {
            ctx.clearRect(0, 0, width, height);
            ctx.drawImage(video, 0, 0, width, height);
          }

          if (now - lastHandCheck > 55) {
            lastHandCheck = now;
            const detectedHands = handLandmarker.detectForVideo(video, now).landmarks;
            if (detectedHands.length) {
              const previousHands = handsRef.current;
              handsRef.current = detectedHands.map((hand, handIndex) => hand.map((point, pointIndex) => {
                const previous = previousHands[handIndex]?.[pointIndex];
                if (!previous) return { x: point.x, y: point.y, z: point.z };
                return { x: previous.x * 0.5 + point.x * 0.5, y: previous.y * 0.5 + point.y * 0.5, z: previous.z * 0.5 + point.z * 0.5 };
              }));
              lastHandSeenAt = now;
              const pinching = detectedHands.some((hand) => {
                const [wrist, thumb, index, middleKnuckle] = [hand[0], hand[4], hand[8], hand[9]];
                const pinchDistance = Math.hypot(thumb.x - index.x, thumb.y - index.y, thumb.z - index.z);
                const palmSize = Math.max(0.04, Math.hypot(wrist.x - middleKnuckle.x, wrist.y - middleKnuckle.y));
                return pinchDistance / palmSize < 0.34;
              });
              setIsPinching(pinching);
              if (pinching && !pinchLatchedRef.current && now - lastPinchAtRef.current > 800) {
                lastPinchAtRef.current = now;
                if (plateReadyRef.current) toggleGhost();
                else beginPlateCountdown();
              }
              pinchLatchedRef.current = pinching;
            } else if (now - lastHandSeenAt > 750) {
              handsRef.current = [];
              pinchLatchedRef.current = false;
              setIsPinching(false);
            }
          }

          drawHandHud(ctx, handsRef.current, width, height);
          setHandCount((current) => current === handsRef.current.length ? current : handsRef.current.length);
          const roundedPercent = Math.round(ghostAmountRef.current * 100);
          if (roundedPercent !== lastPercent) {
            lastPercent = roundedPercent;
            setGhostPercent(roundedPercent);
          }
          frameCount += 1;
          if (now - fpsStartedAt >= 1000) {
            setFps(Math.round((frameCount * 1000) / (now - fpsStartedAt)));
            frameCount = 0;
            fpsStartedAt = now;
          }
        }
      }
      animationFrameId = requestAnimationFrame(renderFrame);
    };

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
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
        setError("Camera access is required for Ghost Mode.");
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
  }, [beginPlateCountdown, hasStarted, isModelLoaded, toggleGhost]);

  useEffect(() => {
    if (!hasStarted) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      if (event.code === "Space" && !plateReadyRef.current) {
        event.preventDefault();
        capturePlate();
      } else if (event.key.toLowerCase() === "b") resetPlate();
      else if (event.key.toLowerCase() === "g") toggleGhost();
      else if (event.key.toLowerCase() === "h") toggleHealing();
      else if (event.key.toLowerCase() === "f") void containerRef.current?.requestFullscreen();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [capturePlate, hasStarted, resetPlate, toggleGhost, toggleHealing]);

  useEffect(() => () => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    void audioContextRef.current?.close();
  }, []);

  const startExperience = () => {
    // Create audio during the click gesture so later pinch-triggered sounds are allowed.
    audioContextRef.current ??= new AudioContext();
    resetPlate();
    setHasStarted(true);
  };
  const stopExperience = () => {
    setHasStarted(false);
    handsRef.current = [];
    pinchLatchedRef.current = false;
    setIsPinching(false);
    setHandCount(0);
    setFps(0);
    resetPlate();
  };
  const captureScreen = () => {
    const canvas = outputRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `ghost-mode-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const statusText = !plateReadyRef.current ? "CAPTURE A CLEAN PLATE" : phase === "vanished" ? "GHOST ACTIVE" : "PORTAL READY · PINCH TO VANISH";

  return (
    <div className="ghost-shell">
      <div ref={containerRef} className={`filter-container ghost-container ghost-container--${phase}`}>
        {!isModelLoaded && !error && <div className="loading-overlay ghost-loading"><div className="spinner" /><p>Loading segmentation + hand tracking…</p></div>}
        {error && <div className="loading-overlay ghost-loading"><div className="ghost-error-mark">!</div><p>{error}</p></div>}
        {isModelLoaded && !hasStarted && !error && (
          <div className="loading-overlay start-screen ghost-start">
            <span className="ghost-orb" aria-hidden="true" />
            <div><span className="mode-card__eyebrow">HAND LAB · GHOST MODE</span><h2>Disappear on command.</h2><p>Capture an empty background, return to frame, then pinch either hand to vanish while your tracked hand skeleton remains.</p></div>
            <button className="btn-primary btn-ghost" onClick={startExperience}>Start camera</button>
          </div>
        )}
        <div className="camera-wrapper">
          <video ref={videoRef} autoPlay playsInline muted className="ghost-source-video" />
          <canvas ref={outputRef} className="camera-canvas ghost-output" />
          {hasStarted && <>
            <div className="ghost-hud">
              <div className={`pinch-indicator ${isPinching ? "is-pinching" : ""}`}><span className="pinch-dot" />{isPinching ? "PINCH DETECTED" : `HANDS ${handCount}`}</div>
              <div className={`ghost-status ghost-status--${phase}`}>{statusText}<small>{fps} FPS</small></div>
            </div>
            {!plateReadyRef.current && <div className="plate-guide">
              {countdown === null ? <><span className="plate-guide__icon">⌁</span><strong>Pinch, then step out of frame</strong><span>The empty room will be captured after three seconds.</span><button className="text-action" onClick={beginPlateCountdown}>Start countdown</button></> : <><span className="countdown-number">{countdown}</span><strong>Step out of frame</strong><span>Space captures the plate early.</span></>}
            </div>}
            {plateReadyRef.current && <div className="ghost-meter" aria-label={`Ghost effect ${ghostPercent}%`}><span>GHOST {ghostPercent}%</span><div><i style={{ width: `${ghostPercent}%` }} /></div></div>}
            <div className="ghost-controls">
              {plateReadyRef.current && <><button className="control-button" onClick={toggleGhost}>{phase === "vanished" ? "Reappear" : "Vanish"}</button><button className="control-button" onClick={captureScreen}>Capture</button></>}
              <button className={`control-button ${healing ? "is-active" : ""}`} onClick={toggleHealing} title="Slowly correct background exposure using person-free pixels">Healing {healing ? "on" : "off"}</button>
              <button className="control-button" onClick={resetPlate}>New plate</button>
              <button className="control-button control-button--stop" onClick={stopExperience}>Stop</button>
            </div>
          </>}
        </div>
      </div>
      <div className="ghost-steps" aria-label="Ghost Mode keyboard controls">
        <span><b>PINCH / G</b> toggle ghost</span><i /><span><b>B</b> new plate</span><i /><span><b>H</b> healing</span><i /><span><b>F</b> fullscreen</span>
      </div>
    </div>
  );
}
