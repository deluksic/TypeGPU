/**
 * Multi-string vibrating strings (DST-I via GPU FFT, modal evolution, strum interaction).
 *
 * 4 strings tuned in perfect fourths (E-A-D-G bass) derived from a single base frequency.
 * Click/drag across strings to pluck (strum). Superposition on repluck.
 */
import tgpu from "typegpu";
import { defineControls } from "../../common/defineControls.ts";
import { createAudioManager } from "./audio.ts";
import { S } from "./constants.ts";
import { createDst } from "./dst.ts";
import { createInteraction } from "./interaction.ts";
import { createRenderer } from "./rendering.ts";
import { createSimulation, type BaseParams } from "./simulation.ts";

const root = await tgpu.init({
  adapter: { powerPreference: "high-performance" },
});
const device = root.device;
const canvas = document.querySelector("canvas") as HTMLCanvasElement;
const context = root.configureContext({ canvas, alphaMode: "premultiplied" });

const wallTimeOriginMs = performance.now();
function wallTime() {
  return (performance.now() - wallTimeOriginMs) / 1000;
}

let baseParams: BaseParams = {
  fundamentalHz: 41.2,
  stiffnessB: 0.00012,
  dampingGamma: 0.15,
  spectralDamping: 5,
  pickupU: 0.22,
  damped: false,
};

function setDamped(on: boolean) {
  dst.reanchor(wallTime());
  baseParams = { ...baseParams, damped: on };
  simulation.updateStringParams(baseParams);
}

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat) {
    e.preventDefault();
    setDamped(true);
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
    setDamped(false);
  }
});

// 1. DST creates mode buffers (modeY0, modeV0)
const dst = createDst(root);

// 2. Simulation creates stringParams, pluckState, spatialY; binds to mode buffers
const simulation = createSimulation(root, {
  modeY0Buffer: dst.modeY0Buffer,
  modeV0Buffer: dst.modeV0Buffer,
});

// 3. DST's superpose kernel needs stringParams + pluckState from simulation
dst.connect({
  stringParamsBuffer: simulation.stringParamsBuffer,
  pluckStateBuffer: simulation.pluckStateBuffer,
});

// 4. Renderer reads spatialY + stringParams (for per-string thickness)
const renderer = createRenderer(root, canvas, context, {
  spatialYBuffer: simulation.spatialYBuffer,
  stringParamsBuffer: simulation.stringParamsBuffer,
});

// 5. Audio reads mode buffers + per-string params
const audio = createAudioManager(root, {
  modeY0Buffer: dst.modeY0Buffer,
  modeV0Buffer: dst.modeV0Buffer,
  stringParamsBuffer: simulation.stringParamsBuffer,
  pluckStateBuffer: simulation.pluckStateBuffer,
});

simulation.updateStringParams(baseParams);

let audioStarted = false;

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    prevFrameTime = wallTime();
    audioStarted = false;
  }
});

const interaction = createInteraction(canvas, {
  onPluck(stringIndex, u, height) {
    audio.resume();
    const t = wallTime();
    dst.pluck(stringIndex, u, height, t);
    if (!audioStarted) {
      audio.startSession(t);
      audioStarted = true;
    }
  },
});

let prevFrameTime = 0;
let frameId = 0;

function loop() {
  const t = wallTime();
  const tStart = prevFrameTime;
  let tEnd = t;
  if (tEnd <= tStart) {
    tEnd = tStart + 1e-5;
  }

  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();
  simulation.update(computePass, tStart, tEnd);
  computePass.end();

  renderer.draw(encoder);

  device.queue.submit([encoder.finish()]);

  prevFrameTime = t;
  audio.pump();

  frameId = requestAnimationFrame(loop);
}
frameId = requestAnimationFrame(loop);

export const controls = defineControls({
  "Mute (space)": {
    initial: false,
    onToggleChange: (v) => {
      setDamped(v);
    },
  },
  "Fundamental (Hz)": {
    initial: 41.2,
    min: 20,
    max: 500,
    step: 1,
    onSliderChange: (v) => {
      baseParams = { ...baseParams, fundamentalHz: v };
      simulation.updateStringParams(baseParams);
    },
  },
  Stiffness: {
    initial: 0.00012,
    min: 0,
    max: 0.01,
    step: 0.00001,
    onSliderChange: (v) => {
      baseParams = { ...baseParams, stiffnessB: v };
      simulation.updateStringParams(baseParams);
    },
  },
  Damping: {
    initial: 0.15,
    min: 0.01,
    max: 2,
    step: 0.01,
    onSliderChange: (v) => {
      baseParams = { ...baseParams, dampingGamma: v };
      simulation.updateStringParams(baseParams);
    },
  },
  "HF decay (spectral)": {
    initial: 5,
    min: 0,
    max: 20,
    step: 0.5,
    onSliderChange: (v) => {
      baseParams = { ...baseParams, spectralDamping: v };
      simulation.updateStringParams(baseParams);
    },
  },
  "Pickup u": {
    initial: 0.22,
    min: 0.05,
    max: 0.95,
    step: 0.01,
    onSliderChange: (v) => {
      baseParams = { ...baseParams, pickupU: v };
      simulation.updateStringParams(baseParams);
    },
  },
  "Visual Y scale": {
    initial: 1 / (S + 1),
    min: 0.02,
    max: 1 / (S + 1) * 3,
    step: 0.01,
    onSliderChange: (v) => {
      renderer.setYScale(v);
    },
  },
  "Audio gain": {
    initial: 0.25,
    min: 0.25,
    max: 6,
    step: 0.25,
    onSliderChange: (v) => {
      audio.setOutputGain(v);
    },
  },
});

export function onCleanup() {
  cancelAnimationFrame(frameId);
  interaction.destroy();
  audio.destroy();
  renderer.destroy();
  simulation.destroy();
  dst.destroy();
  root.destroy();
  root.device.destroy();
}
