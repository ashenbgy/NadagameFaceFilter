"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
} from "@mediapipe/tasks-vision";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

type GhostPhase = "setup" | "live" | "vanished";

export default function GhostFilter() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const outputRef = useRef<HTMLCanvasElement>(null);
  const plateRef = useRef<HTMLCanvasElement | null>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const personRef = useRef<HTMLCanvasElement | null>(null);
  const pinchLatchedRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const segmenterRef = useRef<ImageSegmenter | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const plateReadyRef = useRef(false);
  const phaseRef = useRef<GhostPhase>("setup");
  const pinchingRef = useRef(false);
  const handLandmarksRef = useRef<Array<{ x: number; y: number; z: number }> | null>(null);
  const lastHandSeenAtRef = useRef(0);
  const lastPinchAtRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [phase, setPhase] = useState<GhostPhase>("setup");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isPinching, setIsPinching] = useState(false);
  const [error, setError] = useState("");

  const setGhostPhase = useCallback((next: GhostPhase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const capturePlate = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    const plate = plateRef.current ?? document.createElement("canvas");
    plate.width = video.videoWidth;
    plate.height = video.videoHeight;
    plate.getContext("2d")?.drawImage(video, 0, 0, plate.width, plate.height);
    plateRef.current = plate;
    plateReadyRef.current = true;
    setGhostPhase("live");
    setCountdown(null);
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

  const handlePinch = useCallback(() => {
    const now = performance.now();
    if (now - lastPinchAtRef.current < 900) return;
    lastPinchAtRef.current = now;

    if (!plateReadyRef.current) {
      beginPlateCountdown();
      return;
    }

    setGhostPhase(phaseRef.current === "vanished" ? "live" : "vanished");
  }, [beginPlateCountdown, setGhostPhase]);

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
            numHands: 1,
            minHandDetectionConfidence: 0.42,
            minHandPresenceConfidence: 0.42,
            minTrackingConfidence: 0.4,
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

    loadModels();
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
    const hands = handLandmarkerRef.current;
    if (!video || !output || !segmenter || !hands) return;
    const activeVideo = video;

    const maskCanvas = document.createElement("canvas");
    const personCanvas = document.createElement("canvas");
    maskRef.current = maskCanvas;
    personRef.current = personCanvas;

    let stopped = false;
    let lastHandCheck = 0;

    const renderFrame = () => {
      if (stopped) return;
      if (activeVideo.readyState >= 2 && activeVideo.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = activeVideo.currentTime;
        const width = activeVideo.videoWidth;
        const height = activeVideo.videoHeight;

        if (width && height) {
          if (output.width !== width || output.height !== height) {
            output.width = width;
            output.height = height;
            personCanvas.width = width;
            personCanvas.height = height;
          }

          const now = performance.now();
          const ctx = output.getContext("2d");
          const personCtx = personCanvas.getContext("2d");

          if (ctx && personCtx) {
            if (!plateReadyRef.current) {
              ctx.clearRect(0, 0, width, height);
              ctx.drawImage(activeVideo, 0, 0, width, height);
            } else {
              const result = segmenter.segmentForVideo(activeVideo, now);
              const confidenceMask = result.confidenceMasks?.[0];
              const plate = plateRef.current;

              if (confidenceMask && plate) {
                const values = confidenceMask.getAsFloat32Array();
                if (
                  maskCanvas.width !== confidenceMask.width ||
                  maskCanvas.height !== confidenceMask.height
                ) {
                  maskCanvas.width = confidenceMask.width;
                  maskCanvas.height = confidenceMask.height;
                }
                const maskCtx = maskCanvas.getContext("2d");
                if (maskCtx) {
                  const image = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
                  for (let i = 0; i < values.length; i += 1) {
                    const alpha = Math.max(0, Math.min(255, (values[i] - 0.08) * 290));
                    const p = i * 4;
                    image.data[p] = 255;
                    image.data[p + 1] = 255;
                    image.data[p + 2] = 255;
                    image.data[p + 3] = alpha;
                  }
                  maskCtx.putImageData(image, 0, 0);

                  personCtx.clearRect(0, 0, width, height);
                  personCtx.drawImage(activeVideo, 0, 0, width, height);
                  personCtx.globalCompositeOperation = "destination-in";
                  personCtx.imageSmoothingEnabled = true;
                  personCtx.drawImage(maskCanvas, 0, 0, width, height);
                  personCtx.globalCompositeOperation = "source-over";

                  ctx.clearRect(0, 0, width, height);
                  ctx.drawImage(plate, 0, 0, width, height);

                  if (phaseRef.current === "live") {
                    ctx.drawImage(personCanvas, 0, 0, width, height);
                  }
                }
              }
            }

            if (now - lastHandCheck > 60) {
              lastHandCheck = now;
              const handResult = hands.detectForVideo(activeVideo, now);
              const landmarks = handResult.landmarks[0];

              if (landmarks) {
                const previous = handLandmarksRef.current;
                handLandmarksRef.current = landmarks.map((point, index) => {
                  const oldPoint = previous?.[index];
                  if (!oldPoint) return { x: point.x, y: point.y, z: point.z };

                  // Smooth small inference jumps without making the overlay laggy.
                  const newWeight = 0.48;
                  const oldWeight = 1 - newWeight;
                  return {
                    x: oldPoint.x * oldWeight + point.x * newWeight,
                    y: oldPoint.y * oldWeight + point.y * newWeight,
                    z: oldPoint.z * oldWeight + point.z * newWeight,
                  };
                });
                lastHandSeenAtRef.current = now;

                const thumb = landmarks[4];
                const index = landmarks[8];
                const wrist = landmarks[0];
                const middleKnuckle = landmarks[9];
                const pinchDistance = Math.hypot(
                  thumb.x - index.x,
                  thumb.y - index.y,
                  thumb.z - index.z
                );
                const palmSize = Math.max(
                  0.04,
                  Math.hypot(wrist.x - middleKnuckle.x, wrist.y - middleKnuckle.y)
                );
                const pinching = pinchDistance / palmSize < 0.42;

                if (pinching !== pinchingRef.current) {
                  pinchingRef.current = pinching;
                  setIsPinching(pinching);
                }
                // Toggle only on the leading edge. A missed tracking frame does
                // not count as a release; an open hand must actually be seen.
                if (pinching && !pinchLatchedRef.current) handlePinch();
                pinchLatchedRef.current = pinching;
              } else if (now - lastHandSeenAtRef.current > 1100) {
                // Keep coordinates visible through brief fingertip occlusions.
                handLandmarksRef.current = null;
                pinchingRef.current = false;
                pinchLatchedRef.current = false;
                setIsPinching(false);
              }
            }

            const trackedHand = handLandmarksRef.current;
            if (trackedHand) {
              ctx.save();
              ctx.strokeStyle = "rgba(105, 255, 226, 0.72)";
              ctx.lineWidth = Math.max(1.5, width / 720);
              ctx.shadowColor = "rgba(55, 255, 218, 0.8)";
              ctx.shadowBlur = 8;

              if (phaseRef.current === "live") {
                for (const connection of HandLandmarker.HAND_CONNECTIONS) {
                  const start = trackedHand[connection.start];
                  const end = trackedHand[connection.end];
                  if (!start || !end) continue;
                  ctx.beginPath();
                  ctx.moveTo(start.x * width, start.y * height);
                  ctx.lineTo(end.x * width, end.y * height);
                  ctx.stroke();
                }
              }

              trackedHand.forEach((landmark, index) => {
                const isPinchPoint = index === 4 || index === 8;
                ctx.beginPath();
                ctx.fillStyle = isPinchPoint ? "#ffffff" : "#62ffe0";
                ctx.arc(
                  landmark.x * width,
                  landmark.y * height,
                  isPinchPoint ? 5 : 2.5,
                  0,
                  Math.PI * 2
                );
                ctx.fill();
              });
              ctx.restore();

              // Counter-mirror text so coordinates stay readable on the mirrored canvas.
              ctx.save();
              ctx.translate(width, 0);
              ctx.scale(-1, 1);
              ctx.font = `600 ${Math.max(11, width / 85)}px monospace`;
              ctx.textBaseline = "bottom";
              [
                { index: 4, label: "THUMB" },
                { index: 8, label: "INDEX" },
              ].forEach(({ index, label }) => {
                const point = trackedHand[index];
                if (!point) return;
                const coordinateText = `${label}  x ${point.x.toFixed(3)}  y ${point.y.toFixed(3)}  z ${point.z.toFixed(3)}`;
                const metrics = ctx.measureText(coordinateText);
                const desiredX = (1 - point.x) * width + 10;
                const desiredY = point.y * height + (label === "THUMB" ? -12 : 24);
                const labelX = Math.max(8, Math.min(desiredX, width - metrics.width - 8));
                const labelY = Math.max(22, Math.min(desiredY, height - 8));
                ctx.fillStyle = "rgba(2, 10, 10, 0.76)";
                ctx.fillRect(labelX - 5, labelY - 16, metrics.width + 10, 20);
                ctx.fillStyle = "#bafff1";
                ctx.fillText(coordinateText, labelX, labelY);
              });
              ctx.restore();
            }
          }
        }
      }
      const animationId = requestAnimationFrame(renderFrame);
      animationFrameId = animationId;
    };

    let animationFrameId = 0;
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
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
        setError("Camera access is required for Ghost Mode.");
      }
    }

    startCamera();
    return () => {
      stopped = true;
      cancelAnimationFrame(animationFrameId);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      activeVideo.srcObject = null;
    };
  }, [handlePinch, hasStarted, isModelLoaded]);

  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, []);

  const startExperience = () => {
    plateReadyRef.current = false;
    setGhostPhase("setup");
    setHasStarted(true);
  };

  const resetPlate = () => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = null;
    plateReadyRef.current = false;
    plateRef.current = null;
    setGhostPhase("setup");
    setCountdown(null);
  };

  const stopExperience = () => {
    setHasStarted(false);
    handLandmarksRef.current = null;
    lastHandSeenAtRef.current = 0;
    pinchingRef.current = false;
    pinchLatchedRef.current = false;
    setIsPinching(false);
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

  const statusText = !plateReadyRef.current
    ? "Pinch to capture a clean background plate"
    : phase === "vanished"
      ? "Ghost · pinch to show live"
      : "Live · pinch to vanish";

  return (
    <div className="ghost-shell">
      <div className={`filter-container ghost-container ghost-container--${phase}`}>
        {!isModelLoaded && !error && (
          <div className="loading-overlay ghost-loading">
            <div className="spinner" />
            <p>Loading segmentation + hand tracking…</p>
          </div>
        )}

        {error && (
          <div className="loading-overlay ghost-loading">
            <div className="ghost-error-mark">!</div>
            <p>{error}</p>
          </div>
        )}

        {isModelLoaded && !hasStarted && !error && (
          <div className="loading-overlay start-screen ghost-start">
            <span className="ghost-orb" aria-hidden="true" />
            <div>
              <span className="mode-card__eyebrow">REAL-TIME PERSON SEGMENTATION</span>
              <h2>Disappear on command.</h2>
              <p>Pinch once for ghost mode. Release and pinch again to restore live video.</p>
            </div>
            <button className="btn-primary btn-ghost" onClick={startExperience}>
              Enter Ghost Mode
            </button>
          </div>
        )}

        <div className="camera-wrapper">
          <video ref={videoRef} autoPlay playsInline muted className="ghost-source-video" />
          <canvas ref={outputRef} className="camera-canvas ghost-output" />

          {hasStarted && (
            <>
              <div className="ghost-hud">
                <div className={`pinch-indicator ${isPinching ? "is-pinching" : ""}`}>
                  <span className="pinch-dot" />
                  {isPinching ? "PINCH DETECTED" : "SHOW YOUR HAND"}
                </div>
                <div className={`ghost-status ghost-status--${phase}`}>{statusText}</div>
              </div>

              {!plateReadyRef.current && (
                <div className="plate-guide">
                  {countdown === null ? (
                    <>
                      <span className="plate-guide__icon">⌁</span>
                      <strong>Pinch, then step out of frame</strong>
                      <span>We’ll capture the empty room in 3 seconds.</span>
                      <button className="text-action" onClick={beginPlateCountdown}>
                        Or capture with a tap
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="countdown-number">{countdown}</span>
                      <strong>Step out of frame</strong>
                      <span>Capturing your background plate…</span>
                    </>
                  )}
                </div>
              )}

              <div className="ghost-controls">
                {plateReadyRef.current && (
                  <button className="control-button" onClick={captureScreen} aria-label="Capture image">
                    Capture
                  </button>
                )}
                <button className="control-button" onClick={resetPlate}>
                  New plate
                </button>
                <button className="control-button control-button--stop" onClick={stopExperience}>
                  Stop
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="ghost-steps" aria-label="Ghost Mode instructions">
        <span><b>01</b> Pinch</span>
        <i />
        <span><b>02</b> Step away</span>
        <i />
        <span><b>03</b> Return + pinch to vanish</span>
      </div>
    </div>
  );
}
