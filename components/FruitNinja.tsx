"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const INDEX_FINGERTIP = 8;
const MIN_SWIPE = 18; // px of fingertip travel per hand check before it counts as a slice
const TRAIL_MS = 180;
const BOMB_DEATH_MS = 1300;
const COMBO_GAP_MS = 250;
const START_DELAY_MS = 700;
const SPLAT_MS = 3500;

const DIFFICULTIES = {
  easy: { label: "Easy", spawn: [1100, 1700], bombChance: 0.06, gravity: 1100, lives: 5, burst: 2 },
  medium: { label: "Medium", spawn: [800, 1300], bombChance: 0.12, gravity: 1400, lives: 3, burst: 3 },
  hard: { label: "Hard", spawn: [550, 950], bombChance: 0.2, gravity: 1750, lives: 3, burst: 4 },
} as const;
type Difficulty = keyof typeof DIFFICULTIES;
const LEVELS = Object.keys(DIFFICULTIES) as Difficulty[];

type Fruit = {
  r: number;
  skin: string;
  skinDark: string;
  flesh: string;
  rind: string;
  juice: string;
  pattern: "melon" | "citrus" | "apple" | "kiwi" | "plum";
};
const FRUITS: Fruit[] = [
  { r: 46, skin: "#4cb05a", skinDark: "#1b5e20", flesh: "#ff3b4f", rind: "#e9f7c9", juice: "#ff2440", pattern: "melon" },
  { r: 34, skin: "#ffab2e", skinDark: "#e06000", flesh: "#ffb347", rind: "#fff1d6", juice: "#ff9800", pattern: "citrus" },
  { r: 32, skin: "#ef4440", skinDark: "#8a1414", flesh: "#fff3d1", rind: "#e8322f", juice: "#fff2b3", pattern: "apple" },
  { r: 30, skin: "#fff04a", skinDark: "#c7a900", flesh: "#fff48a", rind: "#fffbe0", juice: "#fff04d", pattern: "citrus" },
  { r: 30, skin: "#a07c48", skinDark: "#5b4220", flesh: "#7cc242", rind: "#b5d98a", juice: "#9ccc65", pattern: "kiwi" },
  { r: 30, skin: "#9d4cc4", skinDark: "#4a1466", flesh: "#ffcf4a", rind: "#ffe9a8", juice: "#c77dff", pattern: "plum" },
];
const MENU_FRUITS: Record<Difficulty, Fruit> = { easy: FRUITS[0], medium: FRUITS[1], hard: FRUITS[4] };
const BOMB_R = 32;

type Item = { x: number; y: number; vx: number; vy: number; rot: number; vrot: number; r: number; fruit: Fruit | null; sliced: boolean };
type Piece = { x: number; y: number; vx: number; vy: number; rot: number; vrot: number; r: number; cut: number; fruit: Fruit; side: -1 | 1 };
type Particle = { x: number; y: number; vx: number; vy: number; r: number; color: string; born: number; life: number };
type Splat = { x: number; y: number; color: string; born: number; blobs: { dx: number; dy: number; r: number }[] };
type Popup = { text: string; x: number; y: number; born: number };
type Point = { x: number; y: number };
type TrailPoint = Point & { t: number };
type Blade = Point & { angle: number; trail: TrailPoint[]; slicing: boolean; combo: number; comboAt: number };
type HandStatus = "loading" | "ready" | "unavailable";
type Phase = "menu" | "play";
type Result = { score: number; best: number; level: Difficulty };
type SessionScore = { id: number; score: number; level: Difficulty };

function readBest(level: Difficulty) {
  try {
    return Number(localStorage.getItem(`fruit-ninja-best-${level}`)) || 0;
  } catch {
    return 0;
  }
}

function writeBest(level: Difficulty, best: number) {
  try {
    localStorage.setItem(`fruit-ninja-best-${level}`, String(best));
  } catch {
    // Storage blocked (private mode etc.): best score just won't persist.
  }
}

export default function FruitNinja() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const audioRef = useRef<AudioContext | null>(null);
  const sizeRef = useRef({ W: 900, H: 560, S: 1 });
  const bladesRef = useRef(new Map<string, Blade>());
  const itemsRef = useRef<Item[]>([]);
  const piecesRef = useRef<Piece[]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const splatsRef = useRef<Splat[]>([]);
  const popupsRef = useRef<Popup[]>([]);
  const missMarksRef = useRef<{ x: number; born: number }[]>([]);
  const phaseRef = useRef<Phase>("menu");
  const levelRef = useRef<Difficulty>("medium");
  const pendingStartRef = useRef<{ level: Difficulty; at: number } | null>(null);
  const lastResultRef = useRef<Result | null>(null);
  const playStartedAtRef = useRef(0);
  const nextSpawnAtRef = useRef(0);
  const lastHandCheckRef = useRef(0);
  const bombHitRef = useRef<{ at: number; x: number; y: number } | null>(null);
  const scoreRef = useRef(0);
  const bestRef = useRef(0);
  const missesRef = useRef(0);

  const [phase, setPhase] = useState<Phase>("menu");
  const [handStatus, setHandStatus] = useState<HandStatus>("loading");
  const [cameraFailed, setCameraFailed] = useState(false);
  // Scores from this visit only; cleared on reload.
  const [sessionScores, setSessionScores] = useState<SessionScore[]>([]);

  // Load the hand tracking model once.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.48,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        handLandmarkerRef.current = landmarker;
        setHandStatus("ready");
      } catch (err) {
        console.error("Unable to load hand tracking model", err);
        setHandStatus("unavailable");
      }
    })();
    return () => {
      cancelled = true;
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
    };
  }, []);

  // Camera runs while this tab is open so the menu can be sliced by hand too.
  useEffect(() => {
    let stopped = false;
    let stream: MediaStream | null = null;
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    videoRef.current = video;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (stopped) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        video.srcObject = stream;
        await video.play();
      } catch (err) {
        console.error("Unable to start camera", err);
        setCameraFailed(true);
      }
    })();
    return () => {
      stopped = true;
      stream?.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
      videoRef.current = null;
    };
  }, []);

  function playNoise(duration: number, type: BiquadFilterType, frequency: number, volume: number) {
    try {
      const ac = audioRef.current ?? new AudioContext();
      audioRef.current = ac;
      if (ac.state === "suspended") void ac.resume();
      const length = Math.floor(ac.sampleRate * duration);
      const buffer = ac.createBuffer(1, length, ac.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < length; i += 1) data[i] = (Math.random() * 2 - 1) * (1 - i / length) ** 2;
      const source = ac.createBufferSource();
      const filter = ac.createBiquadFilter();
      const gain = ac.createGain();
      source.buffer = buffer;
      filter.type = type;
      filter.frequency.value = frequency;
      gain.gain.value = volume;
      source.connect(filter).connect(gain).connect(ac.destination);
      source.start();
    } catch {
      // Audio is a nice-to-have.
    }
  }

  function menuLayout(now: number) {
    const { W, H, S } = sizeRef.current;
    return LEVELS.map((level, i) => ({
      level,
      fruit: MENU_FRUITS[level],
      x: W * (0.25 + i * 0.25),
      y: H * 0.6 + Math.sin(now / 500 + i * 2) * 6,
      r: MENU_FRUITS[level].r * S * 1.2,
      rot: now / 1400 + i,
    }));
  }

  function spawnItem() {
    const { W, H, S } = sizeRef.current;
    const config = DIFFICULTIES[levelRef.current];
    const isBomb = Math.random() < config.bombChance;
    const fruit = isBomb ? null : FRUITS[Math.floor(Math.random() * FRUITS.length)];
    const r = (fruit?.r ?? BOMB_R) * S;
    const x = r + 20 + Math.random() * (W - 2 * r - 40);
    // Launch from below the screen, drifting toward the middle, peaking 50–85% of the way up.
    const peak = H * (0.5 + Math.random() * 0.35);
    itemsRef.current.push({
      x,
      y: H + r,
      vx: ((W / 2 - x) / W) * (200 + Math.random() * 300),
      vy: -Math.sqrt(2 * config.gravity * (peak + r)),
      rot: Math.random() * Math.PI * 2,
      vrot: (Math.random() - 0.5) * 5,
      r,
      fruit,
      sliced: false,
    });
  }

  function burst(x: number, y: number, colors: string[], count: number, speed: number) {
    const now = performance.now();
    for (let i = 0; i < count; i += 1) {
      const a = Math.random() * Math.PI * 2;
      const v = speed * (0.3 + Math.random() * 0.7);
      particlesRef.current.push({
        x,
        y,
        vx: Math.cos(a) * v,
        vy: Math.sin(a) * v - 150,
        r: 2 + Math.random() * 4,
        color: colors[i % colors.length],
        born: now,
        life: 500 + Math.random() * 500,
      });
    }
  }

  // Split a fruit into two halves along the swipe, with juice spray, a wall stain and a splat sound.
  function splitFruit(fruit: Fruit, x: number, y: number, r: number, vx: number, vy: number, from: Point, to: Point) {
    const cut = Math.atan2(to.y - from.y, to.x - from.x);
    for (const side of [-1, 1] as const) {
      piecesRef.current.push({
        x,
        y,
        vx: vx - side * Math.sin(cut) * 160,
        vy: Math.min(vy, 0) * 0.5 + side * Math.cos(cut) * 160,
        rot: 0,
        vrot: side * (1.5 + Math.random() * 2.5),
        r,
        cut,
        fruit,
        side,
      });
    }
    burst(x, y, [fruit.juice, fruit.juice, fruit.flesh], 26, 520);
    const blobs = Array.from({ length: 9 }, () => ({
      dx: (Math.random() - 0.5) * r * 2.4,
      dy: (Math.random() - 0.5) * r * 2.4,
      r: r * (0.15 + Math.random() * 0.45),
    }));
    splatsRef.current.push({ x, y, color: fruit.juice, born: performance.now(), blobs: [{ dx: 0, dy: 0, r: r * 0.9 }, ...blobs] });
    playNoise(0.16, "bandpass", 2600, 0.5);
    playNoise(0.22, "lowpass", 500, 0.5);
  }

  function sliceThrough(blade: Blade, from: Point, to: Point) {
    if (bombHitRef.current) return;
    const now = performance.now();

    if (phaseRef.current === "menu") {
      if (pendingStartRef.current) return;
      for (const m of menuLayout(now)) {
        if (!segmentHitsCircle(from, to, m.x, m.y, m.r)) continue;
        splitFruit(m.fruit, m.x, m.y, m.r, 0, 0, from, to);
        pendingStartRef.current = { level: m.level, at: now + START_DELAY_MS };
        return;
      }
      return;
    }

    for (const item of itemsRef.current) {
      if (item.sliced || !segmentHitsCircle(from, to, item.x, item.y, item.r)) continue;
      item.sliced = true;
      if (!item.fruit) {
        bombHitRef.current = { at: now, x: item.x, y: item.y };
        burst(item.x, item.y, ["#ffb347", "#ff5f1f", "#fff3b0", "#444"], 70, 950);
        playNoise(1.1, "lowpass", 280, 1);
        return;
      }
      scoreRef.current += 1;
      blade.combo = now - blade.comboAt < COMBO_GAP_MS ? blade.combo + 1 : 1;
      blade.comboAt = now;
      splitFruit(item.fruit, item.x, item.y, item.r, item.vx, item.vy, from, to);
    }
  }

  // Moves a blade (mouse or a hand) to a new point, slicing along the way when allowed.
  function moveBlade(key: string, p: Point, canSlice: (distance: number) => boolean) {
    const existing = bladesRef.current.get(key);
    const blade = existing ?? { ...p, angle: -Math.PI / 4, trail: [], slicing: false, combo: 0, comboAt: 0 };
    const distance = Math.hypot(p.x - blade.x, p.y - blade.y);
    if (distance > 2) blade.angle = Math.atan2(p.y - blade.y, p.x - blade.x);
    blade.slicing = Boolean(existing) && canSlice(distance);
    if (blade.slicing) {
      const now = performance.now();
      if (!blade.trail.length) blade.trail.push({ x: blade.x, y: blade.y, t: now });
      sliceThrough(blade, { x: blade.x, y: blade.y }, p);
      blade.trail.push({ ...p, t: now });
    }
    blade.x = p.x;
    blade.y = p.y;
    bladesRef.current.set(key, blade);
  }

  function startGame(level: Difficulty) {
    const now = performance.now();
    levelRef.current = level;
    itemsRef.current = [];
    missMarksRef.current = [];
    popupsRef.current = [];
    pendingStartRef.current = null;
    bombHitRef.current = null;
    scoreRef.current = 0;
    missesRef.current = 0;
    bestRef.current = readBest(level);
    playStartedAtRef.current = now;
    nextSpawnAtRef.current = now + 600;
    phaseRef.current = "play";
    setPhase("play");
  }

  function endGame() {
    const level = levelRef.current;
    const best = Math.max(scoreRef.current, readBest(level));
    writeBest(level, best);
    lastResultRef.current = { score: scoreRef.current, best, level };
    const entry = { id: Date.now(), score: scoreRef.current, level };
    setSessionScores((prev) => [...prev, entry]);
    bombHitRef.current = null;
    itemsRef.current = [];
    phaseRef.current = "menu";
    setPhase("menu");
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const background = document.createElement("canvas");

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = rect.width;
      const H = rect.height;
      sizeRef.current = { W, H, S: Math.max(0.6, Math.min(1.2, W / 900)) };
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      background.width = W * dpr;
      background.height = H * dpr;
      const bg = background.getContext("2d");
      if (bg) {
        bg.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawWood(bg, W, H);
      }
    };
    resize();
    window.addEventListener("resize", resize);

    const pointerPos = (e: PointerEvent): Point => {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };
    let pressed = false;
    const onDown = (e: PointerEvent) => {
      pressed = true;
      moveBlade("mouse", pointerPos(e), () => false);
    };
    const onMove = (e: PointerEvent) => moveBlade("mouse", pointerPos(e), () => pressed);
    const onUp = () => {
      pressed = false;
    };
    const onLeave = () => bladesRef.current.delete("mouse");
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    window.addEventListener("pointerup", onUp);

    let raf = 0;
    let lastTs = 0;
    const step = (ts: number) => {
      raf = requestAnimationFrame(step);
      const dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.1) : 0;
      lastTs = ts;
      const { W, H, S } = sizeRef.current;
      const config = DIFFICULTIES[levelRef.current];
      const playing = phaseRef.current === "play";
      const bomb = bombHitRef.current;

      if (bomb && ts - bomb.at > BOMB_DEATH_MS) endGame();
      const pending = pendingStartRef.current;
      if (pending && ts >= pending.at) startGame(pending.level);

      // Waves get quicker and bigger the longer you survive.
      if (playing && !bomb && ts >= nextSpawnAtRef.current) {
        const elapsed = (ts - playStartedAtRef.current) / 1000;
        const count = 1 + Math.floor(Math.random() * Math.min(config.burst, 1 + elapsed / 20));
        for (let i = 0; i < count; i += 1) spawnItem();
        const [min, max] = config.spawn;
        nextSpawnAtRef.current = ts + (min + Math.random() * (max - min)) * Math.max(0.55, 1 - elapsed / 150);
      }

      // Physics. Whole fruit freezes while a bomb goes off, like the original.
      const moving = bomb ? [...piecesRef.current, ...particlesRef.current] : [...itemsRef.current, ...piecesRef.current, ...particlesRef.current];
      for (const body of moving) {
        body.vy += config.gravity * dt;
        body.x += body.vx * dt;
        body.y += body.vy * dt;
      }
      if (!bomb) for (const item of itemsRef.current) item.rot += item.vrot * dt;
      for (const piece of piecesRef.current) piece.rot += piece.vrot * dt;

      // Each fruit that falls back off the bottom unsliced is one miss.
      let missed = 0;
      itemsRef.current = itemsRef.current.filter((item) => {
        if (item.sliced) return false;
        const gone = item.vy > 0 && item.y - item.r > H;
        if (gone && item.fruit) {
          missed += 1;
          missMarksRef.current.push({ x: item.x, born: ts });
        }
        return !gone;
      });
      if (playing && missed && !bomb) {
        missesRef.current += missed;
        playNoise(0.3, "lowpass", 180, 0.5);
        if (missesRef.current >= config.lives) endGame();
      }
      piecesRef.current = piecesRef.current.filter((p) => p.y - p.r < H + 40);
      particlesRef.current = particlesRef.current.filter((p) => ts - p.born < p.life);
      splatsRef.current = splatsRef.current.filter((s) => ts - s.born < SPLAT_MS);
      popupsRef.current = popupsRef.current.filter((p) => ts - p.born < 1100);
      missMarksRef.current = missMarksRef.current.filter((m) => ts - m.born < 900);

      // Hand tracking: each detected index fingertip is its own sword.
      const video = videoRef.current;
      const handLandmarker = handLandmarkerRef.current;
      const videoReady = Boolean(video && video.readyState >= 2 && video.videoWidth);
      if (video && videoReady && handLandmarker && ts - lastHandCheckRef.current > 40) {
        lastHandCheckRef.current = ts;
        const hands = handLandmarker.detectForVideo(video, ts).landmarks;
        const seen = new Set<string>();
        hands.forEach((hand, i) => {
          const tip = hand[INDEX_FINGERTIP];
          const key = `hand${i}`;
          seen.add(key);
          // Only a real swipe slices: ignore a resting hand, and huge jumps (MediaPipe can swap hand indices).
          moveBlade(key, { x: (1 - tip.x) * W, y: tip.y * H }, (d) => d > MIN_SWIPE && d < W * 0.4);
        });
        bladesRef.current.forEach((_, key) => {
          if (key.startsWith("hand") && !seen.has(key)) bladesRef.current.delete(key);
        });
      }

      // Combos: 3+ fruit in one swipe award a bonus of one point per fruit.
      bladesRef.current.forEach((blade) => {
        if (!blade.combo || ts - blade.comboAt < COMBO_GAP_MS) return;
        if (blade.combo >= 3 && playing) {
          scoreRef.current += blade.combo;
          popupsRef.current.push({ text: `${blade.combo} FRUIT COMBO +${blade.combo}`, x: blade.x, y: blade.y, born: ts });
        }
        blade.combo = 0;
      });

      // ---- Draw ----
      ctx.globalAlpha = 1;
      ctx.drawImage(background, 0, 0, W, H);

      for (const splat of splatsRef.current) {
        ctx.globalAlpha = 0.75 * (1 - (ts - splat.born) / SPLAT_MS);
        ctx.fillStyle = splat.color;
        for (const blob of splat.blobs) {
          ctx.beginPath();
          ctx.arc(splat.x + blob.dx, splat.y + blob.dy, blob.r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;

      if (!playing) {
        const titleSize = Math.max(34, Math.min(84, W * 0.09));
        drawOutlinedText(ctx, "FRUIT NINJA", W / 2, H * 0.17, titleSize, "#ffcc33");
        const result = lastResultRef.current;
        if (result) {
          drawOutlinedText(ctx, `GAME OVER · SCORE ${result.score} · BEST ${result.best}`, W / 2, H * 0.17 + titleSize * 0.85, 20 * S, "#ffffff");
        }
        drawOutlinedText(ctx, result ? "Swipe a fruit to play again" : "Swipe a fruit to start", W / 2, H * 0.17 + titleSize * (result ? 1.35 : 0.9), 18 * S, "#ffe9a8");
        if (!pendingStartRef.current) {
          for (const m of menuLayout(ts)) {
            ctx.save();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
            ctx.lineWidth = 3;
            ctx.setLineDash([12, 9]);
            ctx.lineDashOffset = -ts / 30;
            ctx.beginPath();
            ctx.arc(m.x, m.y, m.r + 16 * S, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
            drawFruit(ctx, m.fruit, m.x, m.y, m.r, m.rot);
            drawOutlinedText(ctx, DIFFICULTIES[m.level].label.toUpperCase(), m.x, m.y + m.r + 40 * S, 22 * S, "#ffffff");
          }
        }
      }

      for (const item of itemsRef.current) {
        if (item.fruit) drawFruit(ctx, item.fruit, item.x, item.y, item.r, item.rot);
        else drawBomb(ctx, item.x, item.y, item.r, item.rot);
      }
      for (const piece of piecesRef.current) drawHalf(ctx, piece);

      for (const p of particlesRef.current) {
        ctx.globalAlpha = 1 - (ts - p.born) / p.life;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      for (const mark of missMarksRef.current) {
        ctx.globalAlpha = 1 - (ts - mark.born) / 900;
        drawX(ctx, Math.max(24, Math.min(W - 24, mark.x)), H - 30, 22, true);
      }
      ctx.globalAlpha = 1;

      for (const popup of popupsRef.current) {
        const age = ts - popup.born;
        ctx.globalAlpha = Math.min(1, (1100 - age) / 300);
        drawOutlinedText(ctx, popup.text, Math.max(120, Math.min(W - 120, popup.x)), popup.y - age * 0.04, 24 * S, "#ffd23f");
      }
      ctx.globalAlpha = 1;

      if (playing) {
        drawFruit(ctx, FRUITS[0], 30, 34, 16, -0.5);
        drawOutlinedText(ctx, String(scoreRef.current), 56, 36, 40, "#ffd23f", "left");
        drawOutlinedText(ctx, `BEST: ${Math.max(bestRef.current, scoreRef.current)} · ${config.label.toUpperCase()}`, 16, 76, 16, "#ffe9a8", "left");
        // Lives as X marks (small → big), turning red as fruit is missed.
        let x = W - 20;
        for (let i = config.lives - 1; i >= 0; i -= 1) {
          const size = 16 + i * (18 / Math.max(1, config.lives - 1));
          x -= size / 2;
          drawX(ctx, x, 34, size, i < missesRef.current);
          x -= size / 2 + 10;
        }
      }

      // Small mirrored camera preview so players can see their hand.
      if (video && videoReady) {
        const pw = Math.min(200, W * 0.24);
        const ph = (pw * video.videoHeight) / video.videoWidth;
        const px = W - pw - 12;
        const py = H - ph - 12;
        ctx.save();
        ctx.translate(px + pw, py);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, pw, ph);
        ctx.restore();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
        ctx.lineWidth = 2;
        ctx.strokeRect(px, py, pw, ph);
      }

      bladesRef.current.forEach((blade) => {
        blade.trail = blade.trail.filter((p) => ts - p.t < TRAIL_MS);
        drawTrail(ctx, blade.trail, ts);
        drawSword(ctx, blade.x, blade.y, blade.angle, blade.slicing && blade.trail.length > 1);
      });

      if (bomb) {
        const t = (ts - bomb.at) / BOMB_DEATH_MS;
        const flash = ctx.createRadialGradient(bomb.x, bomb.y, 0, bomb.x, bomb.y, Math.max(W, H) * (0.3 + t));
        flash.addColorStop(0, `rgba(255, 255, 255, ${1 - t})`);
        flash.addColorStop(1, `rgba(255, 255, 255, ${0.6 * (1 - t)})`);
        ctx.fillStyle = flash;
        ctx.fillRect(0, 0, W, H);
      }
    };
    raf = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      window.removeEventListener("pointerup", onUp);
      void audioRef.current?.close();
      audioRef.current = null;
    };
  }, []);

  return (
    <div className="fruit-shell">
      <div className="fruit-stage">
        <canvas ref={canvasRef} className="fruit-canvas" aria-label="Fruit Ninja game" />
      </div>
      <div className="fruit-hud">
        <span>
          {cameraFailed || handStatus === "unavailable"
            ? "🖱️ Camera unavailable: drag with mouse or touch to slice"
            : handStatus === "ready"
              ? "✋ Hand tracking on: swipe your index finger (mouse works too)"
              : "✋ Loading hand tracking… (mouse works now)"}
        </span>
        {phase === "menu" && (
          <div className="fruit-modes">
            {LEVELS.map((level) => (
              <button key={level} className={`btn-fruit btn-fruit--${level}`} onClick={() => startGame(level)}>
                {DIFFICULTIES[level].label}
              </button>
            ))}
          </div>
        )}
      </div>
      {sessionScores.length > 0 && (
        <div className="fruit-scoreboard">
          <h3>Session scoreboard</h3>
          <ol>
            {[...sessionScores]
              .sort((a, b) => b.score - a.score)
              .slice(0, 5)
              .map((entry) => (
                <li key={entry.id} className={entry.id === sessionScores[sessionScores.length - 1].id ? "is-latest" : ""}>
                  <strong>{entry.score}</strong>
                  <span>{DIFFICULTIES[entry.level].label}</span>
                  {entry.id === sessionScores[sessionScores.length - 1].id && <em>latest</em>}
                </li>
              ))}
          </ol>
          <small>{sessionScores.length} game{sessionScores.length === 1 ? "" : "s"} played · resets when you reload</small>
        </div>
      )}
    </div>
  );
}

// Distance from the fruit's centre to segment a-b, used to detect a slice through its hit circle.
function segmentHitsCircle(a: Point, b: Point, cx: number, cy: number, r: number) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq === 0 ? 0 : ((cx - a.x) * dx + (cy - a.y) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const px = a.x + t * dx;
  const py = a.y + t * dy;
  return (px - cx) ** 2 + (py - cy) ** 2 <= r * r;
}

function circle(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
}

function drawWood(ctx: CanvasRenderingContext2D, W: number, H: number) {
  const planks = 6;
  const pw = W / planks;
  const shades = ["#7a4a28", "#6e4223", "#835230", "#744626", "#6a3f21", "#7d4c2a"];
  for (let i = 0; i < planks; i += 1) {
    const x = i * pw;
    ctx.fillStyle = shades[i];
    ctx.fillRect(x, 0, pw + 1, H);
    ctx.strokeStyle = "rgba(40, 20, 5, 0.2)";
    ctx.lineWidth = 1.5;
    for (let k = 0; k < 7; k += 1) {
      const gx = x + ((k + 0.5) * pw) / 7;
      const wobble = () => (Math.random() - 0.5) * 16;
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.bezierCurveTo(gx + wobble(), H * 0.33, gx + wobble(), H * 0.66, gx + wobble(), H);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(45, 22, 8, 0.45)";
    ctx.beginPath();
    ctx.ellipse(x + pw * (0.3 + Math.random() * 0.4), H * Math.random(), pw * 0.08, pw * 0.16, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(20, 10, 2, 0.6)";
    ctx.fillRect(x - 1.5, 0, 3, H);
  }
  const vignette = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.3, W / 2, H / 2, Math.max(W, H) * 0.75);
  vignette.addColorStop(0, "rgba(0, 0, 0, 0)");
  vignette.addColorStop(1, "rgba(0, 0, 0, 0.5)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, W, H);
}

function drawFruit(ctx: CanvasRenderingContext2D, fruit: Fruit, x: number, y: number, r: number, rot: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  const skin = ctx.createRadialGradient(-r * 0.35, -r * 0.4, r * 0.1, 0, 0, r);
  skin.addColorStop(0, fruit.skin);
  skin.addColorStop(1, fruit.skinDark);
  ctx.fillStyle = skin;
  circle(ctx, 0, 0, r);
  ctx.fill();
  if (fruit.pattern === "melon") {
    ctx.save();
    ctx.clip();
    ctx.fillStyle = fruit.skinDark;
    for (let i = -3; i <= 3; i += 1) {
      ctx.beginPath();
      ctx.ellipse(i * r * 0.32, 0, r * 0.08, r * 1.05, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  if (fruit.pattern === "apple") {
    ctx.strokeStyle = "#5a3515";
    ctx.lineWidth = Math.max(2, r * 0.09);
    ctx.beginPath();
    ctx.moveTo(0, -r * 0.8);
    ctx.lineTo(r * 0.1, -r * 1.2);
    ctx.stroke();
    ctx.fillStyle = "#4caf50";
    ctx.beginPath();
    ctx.ellipse(r * 0.32, -r * 1.05, r * 0.28, r * 0.12, -0.5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (fruit.pattern === "citrus") {
    ctx.fillStyle = "#5b8c2a";
    circle(ctx, 0, -r * 0.9, r * 0.1);
    ctx.fill();
  }
  ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
  ctx.beginPath();
  ctx.ellipse(-r * 0.35, -r * 0.42, r * 0.28, r * 0.15, -0.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// The flesh you see on a cut half: rind ring, flesh, and seeds/segments by fruit type.
function drawCutFace(ctx: CanvasRenderingContext2D, fruit: Fruit, r: number) {
  ctx.fillStyle = fruit.rind;
  circle(ctx, 0, 0, r * 0.9);
  ctx.fill();
  ctx.fillStyle = fruit.flesh;
  circle(ctx, 0, 0, r * (fruit.pattern === "melon" ? 0.76 : 0.8));
  ctx.fill();
  if (fruit.pattern === "melon") {
    ctx.fillStyle = "#1d1d1d";
    for (let i = 0; i < 10; i += 1) {
      const a = (i / 10) * Math.PI * 2;
      ctx.beginPath();
      ctx.ellipse(Math.cos(a) * r * 0.45, Math.sin(a) * r * 0.45, r * 0.07, r * 0.04, a, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (fruit.pattern === "citrus") {
    ctx.strokeStyle = fruit.rind;
    ctx.lineWidth = 2;
    for (let i = 0; i < 10; i += 1) {
      const a = (i / 10) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(Math.cos(a) * r * 0.8, Math.sin(a) * r * 0.8);
      ctx.stroke();
    }
  } else if (fruit.pattern === "apple") {
    ctx.fillStyle = "#f1dca0";
    ctx.beginPath();
    ctx.ellipse(0, 0, r * 0.3, r * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#5a3515";
    for (const sx of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(sx * r * 0.12, 0, r * 0.07, r * 0.04, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (fruit.pattern === "kiwi") {
    ctx.fillStyle = "#f4f9e8";
    circle(ctx, 0, 0, r * 0.24);
    ctx.fill();
    ctx.fillStyle = "#111";
    for (let i = 0; i < 18; i += 1) {
      const a = (i / 18) * Math.PI * 2;
      circle(ctx, Math.cos(a) * r * 0.36, Math.sin(a) * r * 0.36, r * 0.035);
      ctx.fill();
    }
  } else {
    ctx.fillStyle = "#9b5a1f";
    ctx.beginPath();
    ctx.ellipse(0, 0, r * 0.28, r * 0.2, 0, 0, Math.PI * 2);
    ctx.fill();
  }
}

// One half of a sliced fruit, clipped to one side of the cut line.
function drawHalf(ctx: CanvasRenderingContext2D, piece: Piece) {
  const s = piece.r * 1.3;
  ctx.save();
  ctx.translate(piece.x, piece.y);
  ctx.rotate(piece.cut + piece.rot);
  ctx.beginPath();
  ctx.rect(-s, piece.side < 0 ? -s : 0, s * 2, s);
  ctx.clip();
  drawFruit(ctx, piece.fruit, 0, 0, piece.r, 0);
  drawCutFace(ctx, piece.fruit, piece.r);
  ctx.restore();
}

function drawBomb(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, rot: number) {
  ctx.save();
  ctx.translate(x, y);
  const glow = ctx.createRadialGradient(0, 0, r * 0.8, 0, 0, r * 1.7);
  glow.addColorStop(0, "rgba(255, 40, 40, 0.6)");
  glow.addColorStop(1, "rgba(255, 40, 40, 0)");
  ctx.fillStyle = glow;
  circle(ctx, 0, 0, r * 1.7);
  ctx.fill();
  ctx.rotate(rot);
  const body = ctx.createRadialGradient(-r * 0.35, -r * 0.35, r * 0.1, 0, 0, r);
  body.addColorStop(0, "#6a6a6a");
  body.addColorStop(1, "#0d0d0d");
  ctx.fillStyle = body;
  circle(ctx, 0, 0, r);
  ctx.fill();
  ctx.fillStyle = "#3a3a3a";
  ctx.fillRect(-r * 0.25, -r * 1.12, r * 0.5, r * 0.25);
  ctx.strokeStyle = "#c8a15a";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, -r * 1.1);
  ctx.quadraticCurveTo(r * 0.3, -r * 1.15, r * 0.35, -r * 1.45);
  ctx.stroke();
  ctx.shadowColor = "#ffcc33";
  ctx.shadowBlur = 14;
  ctx.fillStyle = Math.random() > 0.5 ? "#fff3b0" : "#ffb347";
  circle(ctx, r * 0.35, -r * 1.45, 3 + Math.random() * 4);
  ctx.fill();
  ctx.restore();
}

function drawX(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, missed: boolean) {
  const h = size / 2;
  ctx.save();
  ctx.translate(x, y);
  ctx.lineCap = "round";
  for (const [color, width] of [
    ["#0b1a2a", size * 0.42],
    [missed ? "#ff2a2a" : "#2d6fb3", size * 0.22],
  ] as const) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(-h, -h);
    ctx.lineTo(h, h);
    ctx.moveTo(h, -h);
    ctx.lineTo(-h, h);
    ctx.stroke();
  }
  ctx.restore();
}

function drawOutlinedText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, color: string, align: CanvasTextAlign = "center") {
  ctx.save();
  ctx.font = `900 ${size}px "Arial Black", Impact, sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#2a1405";
  ctx.lineWidth = size * 0.18;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

// Glowing tapered swipe trail behind the sword tip.
function drawTrail(ctx: CanvasRenderingContext2D, trail: TrailPoint[], now: number) {
  if (trail.length < 2) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.shadowColor = "rgba(160, 220, 255, 0.9)";
  ctx.shadowBlur = 16;
  for (let i = 1; i < trail.length; i += 1) {
    const fade = Math.max(0, 1 - (now - trail[i].t) / TRAIL_MS);
    ctx.strokeStyle = `rgba(235, 248, 255, ${fade})`;
    ctx.lineWidth = 2 + 10 * fade;
    ctx.beginPath();
    ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
    ctx.lineTo(trail[i].x, trail[i].y);
    ctx.stroke();
  }
  ctx.restore();
}

// A katana-style sword with its tip at (x, y), pointing along `angle`.
function drawSword(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, glowing: boolean) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  if (glowing) {
    ctx.shadowColor = "rgba(160, 220, 255, 0.9)";
    ctx.shadowBlur = 18;
  }
  const steel = ctx.createLinearGradient(0, -5, 0, 5);
  steel.addColorStop(0, "#ffffff");
  steel.addColorStop(0.5, "#b8c4d4");
  steel.addColorStop(1, "#6b7688");
  ctx.fillStyle = steel;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-18, -5);
  ctx.lineTo(-80, -5);
  ctx.lineTo(-80, 5);
  ctx.lineTo(-14, 4);
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#d4a13a";
  ctx.fillRect(-86, -13, 6, 26);
  ctx.fillStyle = "#3b2416";
  ctx.fillRect(-112, -4, 26, 8);
  ctx.fillStyle = "#d4a13a";
  circle(ctx, -115, 0, 5);
  ctx.fill();
  ctx.restore();
}
