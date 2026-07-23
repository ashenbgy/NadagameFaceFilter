"use client";

import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export default function FaceFilter() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [hasStarted, setHasStarted] = useState(false);
  
  // Refs to hold audio nodes
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameIdRef = useRef<number>(0);

  // Initialize MediaPipe Face Landmarker on mount
  useEffect(() => {
    async function initializeModel() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: "/models/face_landmarker.task",
            delegate: "GPU",
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 5,
        });
        setFaceLandmarker(landmarker);
        setIsModelLoaded(true);
      } catch (error) {
        console.error("Error loading MediaPipe Face Landmarker:", error);
      }
    }
    initializeModel();
    
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      cancelAnimationFrame(animationFrameIdRef.current);
    };
  }, []);

  const startExperience = async () => {
    if (!isModelLoaded || !faceLandmarker) return;

    // 1. Start Audio & Analysis
    const audio = audioRef.current;
    if (audio && !audioContextRef.current) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioContext();
      audioContextRef.current = ctx;
      
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = ctx.createMediaElementSource(audio);
      source.connect(analyser);
      analyser.connect(ctx.destination);
      
      audio.play().catch(e => console.error("Audio play failed", e));
    }

    setHasStarted(true);
  };

  const stopExperience = () => {
    setHasStarted(false);
    
    // Stop audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const captureScreen = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth || canvas.width;
    tempCanvas.height = video.videoHeight || canvas.height;
    const ctx = tempCanvas.getContext("2d");
    if (!ctx) return;

    // Draw the video frame
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    // Draw the filter overlay
    ctx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);

    const dataUrl = tempCanvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.download = "face-filter-capture.png";
    link.href = dataUrl;
    link.click();
  };

  // Run camera and animation loop once started
  useEffect(() => {
    if (!hasStarted || !isModelLoaded || !faceLandmarker) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const maskImage = new Image();
    maskImage.src = "/masks/mask-1.png";

    const processFrame = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const startTimeMs = performance.now();
        const result = faceLandmarker.detectForVideo(video, startTimeMs);

        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
          // Audio Reactive Scaling & Visualizer
          let audioScale = 1.0;
          if (analyserRef.current) {
            const bufferLength = analyserRef.current.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyserRef.current.getByteFrequencyData(dataArray);
            
            let sum = 0;
            for(let i = 0; i < bufferLength; i++) {
              sum += dataArray[i];
            }
            const average = sum / bufferLength;
            audioScale = 1.0 + (average / 255) * 0.3;

            // Draw Background Visualizer
            const visCanvas = visualizerRef.current;
            if (visCanvas) {
              const visCtx = visCanvas.getContext("2d");
              if (visCtx) {
                if (visCanvas.width !== window.innerWidth || visCanvas.height !== window.innerHeight) {
                  visCanvas.width = window.innerWidth;
                  visCanvas.height = window.innerHeight;
                }
                // Motion blur fade effect
                visCtx.fillStyle = 'rgba(5, 5, 5, 0.15)';
                visCtx.fillRect(0, 0, visCanvas.width, visCanvas.height);

                const barWidth = (visCanvas.width / bufferLength) * 2.5;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                  // Make lower frequencies taller
                  const barHeight = (dataArray[i] / 255) * visCanvas.height * 0.6;
                  
                  // Professional elegant gradient from Cyan to Purple/Pink
                  const alpha = 0.15 + (dataArray[i] / 255) * 0.35;
                  const hue = 190 + (i / bufferLength) * 120 + (performance.now() / 50); 
                  visCtx.fillStyle = `hsla(${hue}, 85%, 65%, ${alpha})`;
                  
                  // Draw bars centered vertically or from bottom
                  visCtx.fillRect(x, visCanvas.height - barHeight, barWidth, barHeight);
                  x += barWidth + 1;
                }
              }
            }
          }

          for (const landmarks of result.faceLandmarks) {
            const leftCheek = landmarks[234];
            const rightCheek = landmarks[454];
            const nose = landmarks[1];

            const leftX = leftCheek.x * canvas.width;
            const leftY = leftCheek.y * canvas.height;
            const rightX = rightCheek.x * canvas.width;
            const rightY = rightCheek.y * canvas.height;
            const noseX = nose.x * canvas.width;
            const noseY = nose.y * canvas.height;
            
            const faceWidth = Math.abs(rightX - leftX);
            const maskWidth = faceWidth * 4.0 * audioScale; // Apply audio scale
            
            const aspectRatio = maskImage.width && maskImage.height ? maskImage.width / maskImage.height : 1;
            const maskHeight = maskWidth / aspectRatio;

            // Audio Reactive Shake and Rotate
            const audioIntensity = (audioScale - 1.0) / 0.3; // 0.0 to 1.0 based on volume
            const shakeX = (Math.random() - 0.5) * 25 * audioIntensity;
            const shakeY = (Math.random() - 0.5) * 25 * audioIntensity;
            
            const time = performance.now() / 1000;
            const rotateOffset = Math.sin(time * 15) * 0.2 * audioIntensity;

            const centerX = (leftX + rightX) / 2 + shakeX;
            const centerY = noseY - (maskHeight * 0.1) + shakeY; 

            const angle = Math.atan2(rightY - leftY, rightX - leftX) + rotateOffset;

            if (maskImage.complete && maskImage.naturalWidth > 1) {
              ctx.save();
              ctx.translate(centerX, centerY);
              ctx.rotate(angle);
              ctx.drawImage(
                maskImage,
                -maskWidth / 2,
                -maskHeight / 2,
                maskWidth,
                maskHeight
              );
              ctx.restore();
            } else {
              // Debug points
              ctx.fillStyle = "red";
              ctx.beginPath();
              ctx.arc(noseX, noseY, 5 * audioScale, 0, 2 * Math.PI);
              ctx.fill();

              ctx.fillStyle = "blue";
              ctx.beginPath();
              ctx.arc(leftX, leftY, 5 * audioScale, 0, 2 * Math.PI);
              ctx.fill();

              ctx.beginPath();
              ctx.arc(rightX, rightY, 5 * audioScale, 0, 2 * Math.PI);
              ctx.fill();
              
              ctx.strokeStyle = "white";
              ctx.lineWidth = 2 * audioScale;
              ctx.beginPath();
              ctx.moveTo(leftX, leftY);
              ctx.lineTo(noseX, noseY);
              ctx.lineTo(rightX, rightY);
              ctx.stroke();
            }
          }
        }
      }
      animationFrameIdRef.current = requestAnimationFrame(processFrame);
    };

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          processFrame();
        };
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };

    startCamera();

    return () => {
      cancelAnimationFrame(animationFrameIdRef.current);
      if (video && video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
        video.srcObject = null;
      }
    };
  }, [hasStarted, isModelLoaded, faceLandmarker]);

  return (
    <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
      <canvas ref={visualizerRef} className="visualizer-canvas" />
      <div className="filter-container">
        <audio ref={audioRef} src="/audio/song.mp3" loop crossOrigin="anonymous" />
      
      {!isModelLoaded && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Loading AI Face Model...</p>
        </div>
      )}
      
      {isModelLoaded && !hasStarted && (
        <div className="loading-overlay start-screen">
          <button 
            className="btn-primary"
            onClick={startExperience}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            Start Experience
          </button>
          <p style={{ marginTop: '0.5rem' }}>
            Turn up your volume to see the magic.
          </p>
        </div>
      )}

      <div className="camera-wrapper" style={{ position: 'relative' }}>
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className="camera-video"
        />
        <canvas 
          ref={canvasRef} 
          className="camera-canvas"
        />
        
        {hasStarted && (
          <div style={{ position: 'absolute', bottom: '2rem', left: '0', right: '0', display: 'flex', justifyContent: 'center', gap: '1rem', zIndex: 10 }}>
            <button 
              className="btn-primary" 
              onClick={captureScreen} 
              style={{ backgroundColor: 'rgba(16, 185, 129, 0.9)', backdropFilter: 'blur(4px)', padding: '0.75rem 1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                <circle cx="12" cy="13" r="4"></circle>
              </svg>
              Capture
            </button>
            <button 
              className="btn-primary" 
              onClick={stopExperience} 
              style={{ backgroundColor: 'rgba(239, 68, 68, 0.9)', backdropFilter: 'blur(4px)', padding: '0.75rem 1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              </svg>
              Stop
            </button>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
