"use client";

import { useState } from "react";
import FaceFilter from "./FaceFilter";
import GhostFilter from "./GhostFilter";
import FruitNinja from "./FruitNinja";

type FilterMode = "yaka" | "ghost" | "fruit";

export default function FilterStudio() {
  const [mode, setMode] = useState<FilterMode>("yaka");

  return (
    <section className="filter-studio" aria-label="Face filter studio">
      <div className="mode-picker" role="tablist" aria-label="Choose a face filter">
        <button
          className={`mode-card ${mode === "yaka" ? "is-active" : ""}`}
          onClick={() => setMode("yaka")}
          role="tab"
          aria-selected={mode === "yaka"}
        >
          <span className="mode-card__eyebrow">01 · ORIGINAL</span>
          <strong>YAKA Mask</strong>
          <span>Audio-reactive face tracking</span>
        </button>

        <button
          className={`mode-card mode-card--ghost ${mode === "ghost" ? "is-active" : ""}`}
          onClick={() => setMode("ghost")}
          role="tab"
          aria-selected={mode === "ghost"}
        >
          <span className="mode-card__eyebrow">02 · LIVE AI</span>
          <strong>Ghost Mode</strong>
          <span>Pinch-controlled invisibility</span>
        </button>

        <button
          className={`mode-card mode-card--fruit ${mode === "fruit" ? "is-active" : ""}`}
          onClick={() => setMode("fruit")}
          role="tab"
          aria-selected={mode === "fruit"}
        >
          <span className="mode-card__eyebrow">03 · GAME</span>
          <strong>Fruit Ninja</strong>
          <span>Swipe to slice, dodge bombs</span>
        </button>
      </div>

      <div role="tabpanel" className="mode-stage" key={mode}>
        {mode === "yaka" ? <FaceFilter /> : mode === "ghost" ? <GhostFilter /> : <FruitNinja />}
      </div>
    </section>
  );
}
