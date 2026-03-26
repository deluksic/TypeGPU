import {
  caps,
  endCapSlot,
  LineControlPoint,
  lineSegmentVariableWidth,
  startCapSlot,
} from "@typegpu/geometry";
import tgpu, { d } from "typegpu";
import {
  arrayOf,
  builtin,
  f32,
  i32,
  struct,
  u32,
  vec2f,
  vec3f,
  vec4f,
} from "typegpu/data";
import {
  abs,
  clamp,
  cos,
  cross,
  dot,
  exp,
  floor,
  fract,
  fwidth,
  max,
  min,
  mix,
  neg,
  normalize,
  pow,
  reflect,
  select,
  sin,
  smoothstep,
  sqrt,
  textureSample,
  workgroupBarrier,
} from "typegpu/std";
import {
  AUDIO_SAMPLES_PER_WG,
  AUDIO_WG,
  CHUNK_SAMPLES,
  DAMPER_U_END,
  DAMPER_U_START,
  INTRA_FRAME_SAMPLES,
  L,
  MAX_JOIN,
  N,
  S,
  SPATIAL_Y_FLOATS,
  SPATIAL_Y_PER_STRING,
  SYNTH_SCALE,
  WG,
} from "./constants.ts";

// ---------------------------------------------------------------------------
// Struct types
// ---------------------------------------------------------------------------

export const StringParams = struct({
  omega0: f32,
  stiffnessB: f32,
  dampingGamma: f32,
  spectralDamping: f32,
  pickupU: f32,
  damperStrength: f32,
});

export const PluckState = struct({
  pluckTimeOrigin: f32,
  isActive: u32,
});

export const GlobalUniforms = struct({
  frameTimeStart: f32,
  frameTimeEnd: f32,
  stringCount: u32,
});

export const AudioUniforms = struct({
  chunkStartTime: f32,
  invSampleRate: f32,
  chunkSampleCount: u32,
  stringCount: u32,
});

export const RenderUniform = struct({
  aspect: f32,
  yScale: f32,
  trailLayerAlpha: f32,
  n: u32,
  stringCount: u32,
});

export const LineVertexDrawUniform = struct({
  ndcShift: vec2f,
  radiusScale: f32,
});

export const PluckCommand = struct({
  stringIndex: u32,
  pluckU: f32,
  pluckHeight: f32,
  currentTime: f32,
});

// ---------------------------------------------------------------------------
// Bind group layouts
// ---------------------------------------------------------------------------

export const synthesizeLayout = tgpu.bindGroupLayout({
  globals: { uniform: GlobalUniforms },
  stringParams: { storage: arrayOf(StringParams, S), access: "readonly" },
  pluckState: { storage: arrayOf(PluckState, S), access: "readonly" },
  sinTable: { storage: arrayOf(f32, N * N), access: "readonly" },
  modeY0: { storage: arrayOf(f32, S * N), access: "readonly" },
  modeV0: { storage: arrayOf(f32, S * N), access: "readonly" },
  spatialY: { storage: arrayOf(f32, SPATIAL_Y_FLOATS), access: "mutable" },
});

export const audioLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: AudioUniforms },
  stringParams: { storage: arrayOf(StringParams, S), access: "readonly" },
  pluckState: { storage: arrayOf(PluckState, S), access: "readonly" },
  modeY0: { storage: arrayOf(f32, S * N), access: "readonly" },
  modeV0: { storage: arrayOf(f32, S * N), access: "readonly" },
  audioOut: { storage: arrayOf(f32, CHUNK_SAMPLES), access: "mutable" },
});

export const lineBindLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: RenderUniform },
  lineVertexDraw: { uniform: LineVertexDrawUniform },
  spatialY: { storage: arrayOf(f32, SPATIAL_Y_FLOATS), access: "readonly" },
  stringParams: { storage: arrayOf(StringParams, S), access: "readonly" },
});

export const presentLayout = tgpu.bindGroupLayout({
  scene: { texture: d.texture2d(d.f32) },
  samp: { sampler: "filtering" },
});

export const embedLayout = tgpu.bindGroupLayout({
  cmd: { uniform: PluckCommand },
  fftInput: { storage: arrayOf(vec2f, L), access: "mutable" },
});

export const extractLayout = tgpu.bindGroupLayout({
  fftOutput: { storage: arrayOf(vec2f, L), access: "readonly" },
  deltaModes: { storage: arrayOf(f32, N), access: "mutable" },
});

export const superposeLayout = tgpu.bindGroupLayout({
  cmd: { uniform: PluckCommand },
  stringParams: { storage: arrayOf(StringParams, S), access: "readonly" },
  pluckState: { storage: arrayOf(PluckState, S), access: "mutable" },
  deltaModes: { storage: arrayOf(f32, N), access: "readonly" },
  modeY0: { storage: arrayOf(f32, S * N), access: "mutable" },
  modeV0: { storage: arrayOf(f32, S * N), access: "mutable" },
});

export const reanchorLayout = tgpu.bindGroupLayout({
  currentTime: { uniform: f32 },
  stringParams: { storage: arrayOf(StringParams, S), access: "readonly" },
  pluckState: { storage: arrayOf(PluckState, S), access: "mutable" },
  modeY0: { storage: arrayOf(f32, S * N), access: "mutable" },
  modeV0: { storage: arrayOf(f32, S * N), access: "mutable" },
});

// ---------------------------------------------------------------------------
// Compute kernels
// ---------------------------------------------------------------------------

/** Dispatch (ceil(N/WG), INTRA_FRAME_SAMPLES, S). gid.z = string index. */
export const synthesizeMultiKernel = tgpu.computeFn({
  workgroupSize: [WG, 1, 1],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  "use gpu";
  const j = gid.x;
  const k = gid.y;
  const sIdx = gid.z;
  const n = u32(N);
  const nSamp = u32(INTRA_FRAME_SAMPLES);
  const sCount = synthesizeLayout.$.globals.stringCount;
  if (j >= n || k >= nSamp || sIdx >= sCount) {
    return;
  }

  const state = synthesizeLayout.$.pluckState[sIdx];
  if (state.isActive === u32(0)) {
    synthesizeLayout.$.spatialY[
      sIdx * u32(SPATIAL_Y_PER_STRING) + k * n + j
    ] = f32(0);
    return;
  }

  const t0Global = synthesizeLayout.$.globals.frameTimeStart;
  const t1Global = synthesizeLayout.$.globals.frameTimeEnd;
  const origin = state.pluckTimeOrigin;
  const denom = max(f32(nSamp) - 1.0, 1.0);
  const tGlobal = mix(t0Global, t1Global, f32(k) / denom);
  const t = max(tGlobal - origin, f32(0));

  const params = synthesizeLayout.$.stringParams[sIdx];
  const omega0 = params.omega0;
  const B = params.stiffnessB;
  const gamma = params.dampingGamma;
  const spectral = params.spectralDamping;
  const damper = params.damperStrength;
  const omegaSafe = max(omega0, f32(1));

  const dU1 = f32(DAMPER_U_START);
  const dU2 = f32(DAMPER_U_END);
  const dWidth = dU2 - dU1;
  // oxlint-disable-next-line oxc(approx-constant) -- WGSL π
  const PI = f32(3.141592653589793);

  const base = sIdx * n;
  let sum = f32(0);
  for (let m = u32(0); m < n; m = m + 1) {
    const m1 = f32(m) + 1.0;
    const omega = m1 * omega0 * sqrt(1.0 + B * m1 * m1);
    const safeOmega = max(omega, f32(1e-5));
    const y0 = synthesizeLayout.$.modeY0[base + m];
    const v0 = synthesizeLayout.$.modeV0[base + m];
    const relFreq = max(safeOmega / omegaSafe - 1.0, 0.0);
    const dampMul = 1.0 + (spectral * relFreq * relFreq) / f32(40.0);
    const avgSin2 = f32(0.5) - (sin(2.0 * m1 * PI * dU2) - sin(2.0 * m1 * PI * dU1)) / (4.0 * m1 * PI * dWidth);
    const alpha = gamma * dampMul + damper * avgSin2;
    const dec = exp(-alpha * t);
    const ym =
      dec * (y0 * cos(safeOmega * t) + (v0 / safeOmega) * sin(safeOmega * t));
    const basis = synthesizeLayout.$.sinTable[m * n + j];
    sum = sum + ym * basis;
  }
  synthesizeLayout.$.spatialY[
    sIdx * u32(SPATIAL_Y_PER_STRING) + k * n + j
  ] = sum * f32(SYNTH_SCALE);
});

/**
 * Audio: velocity at pickup, parallelized over strings.
 * Each thread handles one string for one sample (N mode iterations).
 * Workgroup shared memory + barrier to reduce per-string contributions.
 * Dispatch (ceil(CHUNK_SAMPLES / AUDIO_SAMPLES_PER_WG)).
 */
const audioShared = tgpu.workgroupVar(arrayOf(f32, AUDIO_WG));

export const audioKernel = tgpu.computeFn({
  workgroupSize: [AUDIO_WG],
  in: {
    lid: builtin.localInvocationId,
    wid: builtin.workgroupId,
  },
})(({ lid, wid }) => {
  "use gpu";
  const localIdx = lid.x;
  const sIdx = localIdx % u32(S);
  const sampleLocal = u32(f32(localIdx) / f32(S));
  const sampleGlobal = wid.x * u32(AUDIO_SAMPLES_PER_WG) + sampleLocal;

  const n = u32(N);
  const count = audioLayout.$.uniforms.chunkSampleCount;
  const t0 = audioLayout.$.uniforms.chunkStartTime;
  const invSr = audioLayout.$.uniforms.invSampleRate;

  let contribution = f32(0);

  if (sampleGlobal < count && sIdx < u32(S)) {
    const state = audioLayout.$.pluckState[sIdx];
    if (state.isActive !== u32(0)) {
      const tAbs = t0 + f32(sampleGlobal) * invSr;
      const t = max(tAbs - state.pluckTimeOrigin, f32(0));
      const params = audioLayout.$.stringParams[sIdx];
      const omega0 = params.omega0;
      const B = params.stiffnessB;
      const gamma = params.dampingGamma;
      const spectral = params.spectralDamping;
      const uPick = params.pickupU;
      const damper = params.damperStrength;
      const omegaSafe = max(omega0, f32(1));
      const base = sIdx * n;

      const adU1 = f32(DAMPER_U_START);
      const adU2 = f32(DAMPER_U_END);
      const adWidth = adU2 - adU1;
      // oxlint-disable-next-line oxc(approx-constant) -- WGSL π
      const aPI = f32(3.141592653589793);

      let stringSum = f32(0);
      for (let m = u32(0); m < n; m = m + 1) {
        const m1 = f32(m) + 1.0;
        const omega = m1 * omega0 * sqrt(1.0 + B * m1 * m1);
        const safeOmega = max(omega, f32(1e-5));
        const y0 = audioLayout.$.modeY0[base + m];
        const v0 = audioLayout.$.modeV0[base + m];
        const relFreq = max(safeOmega / omegaSafe - 1.0, 0.0);
        const dampMul = 1.0 + (spectral * relFreq * relFreq) / f32(40.0);
        const aAvgSin2 = f32(0.5) - (sin(2.0 * m1 * aPI * adU2) - sin(2.0 * m1 * aPI * adU1)) / (4.0 * m1 * aPI * adWidth);
        const alpha = gamma * dampMul + damper * aAvgSin2;
        const dec = exp(-alpha * t);
        const c = cos(safeOmega * t);
        const s = sin(safeOmega * t);
        const b = v0 / safeOmega;
        const oscill = y0 * c + b * s;
        const ymVel =
          dec * (-alpha * oscill - y0 * safeOmega * s + b * safeOmega * c);
        // oxlint-disable-next-line oxc(approx-constant) -- WGSL π
        const w = sin(3.141592653589793 * uPick * m1);
        stringSum = stringSum + ymVel * w * f32(SYNTH_SCALE);
      }

      contribution = stringSum * f32(2500.0) / (f32(n) * omegaSafe);
    }
  }

  audioShared.$[localIdx] = contribution;
  workgroupBarrier();

  if (sIdx === u32(0) && sampleGlobal < count) {
    let total = f32(0);
    const base = sampleLocal * u32(S);
    for (let s = u32(0); s < u32(S); s = s + 1) {
      total = total + audioShared.$[base + s];
    }
    audioLayout.$.audioOut[sampleGlobal] = total;
  }
});

/** Build odd-extended triangle displacement into FFT input buffer. Dispatch (ceil(L/WG)). */
export const embedKernel = tgpu.computeFn({
  workgroupSize: [WG],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  "use gpu";
  const i = gid.x;
  const n = u32(N);
  const np1 = n + u32(1);
  const lLen = u32(L);
  if (i >= lLen) {
    return;
  }

  const peakU = embedLayout.$.cmd.pluckU;
  const h = embedLayout.$.cmd.pluckHeight;

  let re = f32(0);
  if (i >= u32(1) && i <= n) {
    const u = f32(i) / f32(np1);
    re = select(
      h * (f32(1) - u) / max(f32(1) - peakU, f32(1e-8)),
      h * u / max(peakU, f32(1e-8)),
      u <= peakU,
    );
  } else if (i > np1) {
    const j = lLen - i;
    const u = f32(j) / f32(np1);
    const tent = select(
      h * (f32(1) - u) / max(f32(1) - peakU, f32(1e-8)),
      h * u / max(peakU, f32(1e-8)),
      u <= peakU,
    );
    re = -tent;
  }

  embedLayout.$.fftInput[i] = vec2f(re, f32(0));
});

/** Extract DST-I modes from FFT output: mode[m] = -Im(Z[m+1]) / 2. Dispatch (ceil(N/WG)). */
export const extractKernel = tgpu.computeFn({
  workgroupSize: [WG],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  "use gpu";
  const m = gid.x;
  if (m >= u32(N)) {
    return;
  }
  const z = extractLayout.$.fftOutput[m + u32(1)];
  extractLayout.$.deltaModes[m] = -z.y / f32(2);
});

/**
 * Superpose new pluck onto existing string state.
 * Evaluates current y_m(t), v_m(t), adds deltaY_m, resets time origin.
 * Dispatch (ceil(N/WG)).
 */
export const superposeKernel = tgpu.computeFn({
  workgroupSize: [WG],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  "use gpu";
  const m = gid.x;
  const n = u32(N);
  if (m >= n) {
    return;
  }

  const sIdx = superposeLayout.$.cmd.stringIndex;
  const currentTime = superposeLayout.$.cmd.currentTime;
  const params = superposeLayout.$.stringParams[sIdx];
  const state = superposeLayout.$.pluckState[sIdx];

  const elapsed = max(currentTime - state.pluckTimeOrigin, f32(0));

  const omega0 = params.omega0;
  const B = params.stiffnessB;
  const gamma = params.dampingGamma;
  const spectral = params.spectralDamping;
  const damper = params.damperStrength;

  const m1 = f32(m) + f32(1);
  const omega = m1 * omega0 * sqrt(f32(1) + B * m1 * m1);
  const safeOmega = max(omega, f32(1e-5));
  const omegaSafe = max(omega0, f32(1));
  const relFreq = max(safeOmega / omegaSafe - f32(1), f32(0));
  const dampMul = f32(1) + (spectral * relFreq * relFreq) / f32(40);
  // oxlint-disable-next-line oxc(approx-constant) -- WGSL π
  const sPI = f32(3.141592653589793);
  const sdU1 = f32(DAMPER_U_START);
  const sdU2 = f32(DAMPER_U_END);
  const sdWidth = sdU2 - sdU1;
  const sAvgSin2 = f32(0.5) - (sin(2.0 * m1 * sPI * sdU2) - sin(2.0 * m1 * sPI * sdU1)) / (4.0 * m1 * sPI * sdWidth);
  const alpha = gamma * dampMul + damper * sAvgSin2;

  const base = sIdx * n;
  const Y0 = superposeLayout.$.modeY0[base + m];
  const V0 = superposeLayout.$.modeV0[base + m];

  let yCurrent = f32(0);
  let vCurrent = f32(0);

  if (state.isActive !== u32(0)) {
    if (elapsed > f32(0)) {
      const dec = exp(-alpha * elapsed);
      const c = cos(safeOmega * elapsed);
      const s = sin(safeOmega * elapsed);
      const bCoeff = V0 / safeOmega;
      yCurrent = dec * (Y0 * c + bCoeff * s);
      vCurrent =
        dec *
        ((-alpha * Y0 + V0) * c +
          (-alpha * bCoeff - Y0 * safeOmega) * s);
    } else {
      yCurrent = Y0;
      vCurrent = V0;
    }
  }

  const deltaY = superposeLayout.$.deltaModes[m];
  superposeLayout.$.modeY0[base + m] = yCurrent + deltaY;
  superposeLayout.$.modeV0[base + m] = vCurrent;

  if (m === u32(0)) {
    superposeLayout.$.pluckState[sIdx].pluckTimeOrigin = currentTime;
    superposeLayout.$.pluckState[sIdx].isActive = u32(1);
  }
});

/**
 * Re-anchor all strings: evaluate current y_m(t), v_m(t) (including damper)
 * and write back as new Y0, V0 with updated time origin.
 * Dispatch (ceil(N/WG), S).
 */
export const reanchorKernel = tgpu.computeFn({
  workgroupSize: [WG, 1],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  "use gpu";
  const m = gid.x;
  const sIdx = gid.y;
  const n = u32(N);
  if (m >= n || sIdx >= u32(S)) {
    return;
  }

  const state = reanchorLayout.$.pluckState[sIdx];
  if (state.isActive === u32(0)) {
    return;
  }

  const currentTime = reanchorLayout.$.currentTime;
  const elapsed = max(currentTime - state.pluckTimeOrigin, f32(0));
  if (elapsed <= f32(0)) {
    return;
  }

  const params = reanchorLayout.$.stringParams[sIdx];
  const omega0 = params.omega0;
  const B = params.stiffnessB;
  const gamma = params.dampingGamma;
  const spectral = params.spectralDamping;
  const damper = params.damperStrength;

  const m1 = f32(m) + f32(1);
  const omega = m1 * omega0 * sqrt(f32(1) + B * m1 * m1);
  const safeOmega = max(omega, f32(1e-5));
  const omegaSafe = max(omega0, f32(1));
  const relFreq = max(safeOmega / omegaSafe - f32(1), f32(0));
  const dampMul = f32(1) + (spectral * relFreq * relFreq) / f32(40);
  // oxlint-disable-next-line oxc(approx-constant) -- WGSL π
  const rPI = f32(3.141592653589793);
  const rdU1 = f32(DAMPER_U_START);
  const rdU2 = f32(DAMPER_U_END);
  const rdWidth = rdU2 - rdU1;
  const rAvgSin2 = f32(0.5) - (sin(2.0 * m1 * rPI * rdU2) - sin(2.0 * m1 * rPI * rdU1)) / (4.0 * m1 * rPI * rdWidth);
  const alpha = gamma * dampMul + damper * rAvgSin2;

  const base = sIdx * n;
  const Y0 = reanchorLayout.$.modeY0[base + m];
  const V0 = reanchorLayout.$.modeV0[base + m];

  const dec = exp(-alpha * elapsed);
  const c = cos(safeOmega * elapsed);
  const s = sin(safeOmega * elapsed);
  const bCoeff = V0 / safeOmega;
  reanchorLayout.$.modeY0[base + m] = dec * (Y0 * c + bCoeff * s);
  reanchorLayout.$.modeV0[base + m] =
    dec *
    ((-alpha * Y0 + V0) * c +
      (-alpha * bCoeff - Y0 * safeOmega) * s);

  if (m === u32(0)) {
    reanchorLayout.$.pluckState[sIdx].pluckTimeOrigin = currentTime;
  }
});

// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------

/** y(u) in time-slab `layer` for string `stringIdx`. */
const stringYAtU = tgpu.fn(
  [f32, u32, u32, u32],
  f32,
)((along, n, stringIdx, layer) => {
  "use gpu";
  const buf = lineBindLayout.$.spatialY;
  const base =
    stringIdx * u32(SPATIAL_Y_PER_STRING) + layer * n;
  if (along <= f32(0) || along >= f32(1)) {
    return f32(0);
  }
  const np1 = f32(n + u32(1));
  const tFirst = f32(1.0) / np1;
  const tLast = f32(n) / np1;
  const y0 = buf[base];
  const yLast = buf[base + n - u32(1)];
  if (along < tFirst) {
    return y0 * (along / tFirst);
  }
  if (along > tLast) {
    return yLast * ((f32(1.0) - along) / (f32(1.0) - tLast));
  }
  const s = along * np1;
  const j = u32(floor(s)) - u32(1);
  const tj = f32(j + u32(1)) / np1;
  const tjp = f32(j + u32(2)) / np1;
  const a = (along - tj) / (tjp - tj);
  return mix(buf[base + j], buf[base + j + u32(1)], a);
});

export const lineVertex = tgpu.vertexFn({
  in: {
    instanceIndex: builtin.instanceIndex,
    vertexIndex: builtin.vertexIndex,
  },
  out: {
    outPos: builtin.position,
    uv: vec2f,
  },
})(({ instanceIndex, vertexIndex }) => {
  "use gpu";
  const aspect = lineBindLayout.$.uniforms.aspect;
  const yScale = lineBindLayout.$.uniforms.yScale;
  const n = lineBindLayout.$.uniforms.n;
  const sCount = lineBindLayout.$.uniforms.stringCount;
  const hxIso = f32(0.86);
  const ndcCap = f32(1.0);
  const kIsoToNdc = ndcCap / hxIso;

  const segCount = n + u32(1);
  const layerCount = u32(INTRA_FRAME_SAMPLES);
  const segsPerString = segCount * layerCount;
  const stringIdx = u32(instanceIndex / segsPerString);
  const remainder = instanceIndex - stringIdx * segsPerString;
  const layer = u32(remainder / segCount);
  const seg = remainder - layer * segCount;

  const k = i32(seg);
  const last = i32(n) - 1;
  const iA = u32(clamp(k - 1, 0, last));
  const iB = u32(clamp(k, 0, last));
  const iC = u32(clamp(k + 1, 0, last));
  const iD = u32(clamp(k + 2, 0, last));

  const denomU = max(f32(n) - 1.0, 1.0);
  const uA = f32(iA) / denomU;
  const uB = f32(iB) / denomU;
  const uC = f32(iC) / denomU;
  const uD = f32(iD) / denomU;

  const xA = (uA * 2.0 - 1.0) * hxIso;
  const xB = (uB * 2.0 - 1.0) * hxIso;
  const xC = (uC * 2.0 - 1.0) * hxIso;
  const xD = (uD * 2.0 - 1.0) * hxIso;

  const yA = stringYAtU(uA, n, stringIdx, layer) * yScale;
  const yB = stringYAtU(uB, n, stringIdx, layer) * yScale;
  const yC = stringYAtU(uC, n, stringIdx, layer) * yScale;
  const yD = stringYAtU(uD, n, stringIdx, layer) * yScale;

  const stringOmega = lineBindLayout.$.stringParams[stringIdx].omega0;
  const refOmega = lineBindLayout.$.stringParams[sCount - u32(1)].omega0;
  const radiusFactor = sqrt(max(refOmega, f32(1)) / max(stringOmega, f32(1)));
  const baseRad = f32(0.012) / f32(S);
  const rad = baseRad * lineBindLayout.$.lineVertexDraw.radiusScale * radiusFactor;
  const A = LineControlPoint({ position: vec2f(xA, yA), radius: rad });
  const B = LineControlPoint({ position: vec2f(xB, yB), radius: rad });
  const C = LineControlPoint({ position: vec2f(xC, yC), radius: rad });
  const D = LineControlPoint({ position: vec2f(xD, yD), radius: rad });

  const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D, MAX_JOIN);
  const vIso = result.vertexPosition;

  const stringSpacing = 2.0 / (f32(sCount) + 1.0);
  const stringYOff = 1.0 - stringSpacing * (f32(stringIdx) + 1.0);

  const ndcBase = vec2f(
    kIsoToNdc * vIso.x,
    kIsoToNdc * aspect * vIso.y + stringYOff,
  );
  const ndc = ndcBase + lineBindLayout.$.lineVertexDraw.ndcShift;
  const coreVertexIndex = select(0, (vertexIndex - 2) & 3, vertexIndex >= 2);
  const nearSide = select(
    f32(1),
    f32(-1),
    coreVertexIndex === 1 || coreVertexIndex === 2,
  );
  return {
    outPos: vec4f(ndc * result.w, 0, result.w),
    uv: vec2f(
      vIso.x * f32(0.5) + f32(0.5),
      select(nearSide, f32(0), vertexIndex < 2),
    ),
  };
});

export const lineFragment = tgpu.fragmentFn({
  in: { uv: vec2f },
  out: vec4f,
})(({ uv }) => {
  "use gpu";
  const a = lineBindLayout.$.uniforms.trailLayerAlpha;
  const ribsPerUnit = f32(250);
  const invRibs = f32(1) / ribsPerUnit;
  const halfPeriod = f32(0.5) * invRibs;
  const Ruv = halfPeriod;
  const ribSkew = f32(0.0005);
  const n = (uv.x + ribSkew * uv.y) * ribsPerUnit;
  const sx = (fract(n) - f32(0.5)) * invRibs;
  const u = sx / Ruv;
  const u2 = u * u;
  const onDome = u2 <= f32(1);
  const circleCore = max(f32(1e-10), f32(1) - u2);
  const ribAmp = f32(0.016);
  const h = select(f32(0), ribAmp * sqrt(circleCore), onDome);
  const insideDeriv = u2 < f32(1) - f32(3e-5);
  const dhDu = select(
    f32(0),
    (-ribAmp * u) / sqrt(max(f32(1e-10), f32(1) - u2)),
    insideDeriv,
  );
  const hPrime = dhDu / Ruv;

  const theta = clamp(uv.y, f32(-1), f32(1)) * f32(1.495);
  const st = sin(theta);
  const ct = cos(theta);
  const Sc = f32(1) + h;
  const Pu = vec3f(f32(1), hPrime * st, hPrime * ct);
  const Pv = vec3f(f32(0), Sc * ct, -Sc * st);
  let Nrm = normalize(cross(Pu, Pv));
  const V = vec3f(f32(0), f32(0), f32(1));
  Nrm = select(Nrm, neg(Nrm), dot(Nrm, V) < f32(0));

  const Ldir = normalize(vec3f(f32(0.32), f32(0.65), f32(0.68)));
  const diff = max(f32(0), dot(Nrm, Ldir));
  const Rv = reflect(neg(Ldir), Nrm);
  const spec = pow(max(f32(0), dot(Rv, V)), f32(2)) * f32(0.72);

  const grooveCol = vec3f(0.34, 0.3, 0.26);
  const crestCol = vec3f(0.72, 0.65, 0.58);
  const albedo = mix(grooveCol, crestCol, h / max(ribAmp, f32(1e-4)));

  const rim = min(abs(uv.y), f32(1));
  const rimAtten = mix(f32(1), f32(0.9), smoothstep(f32(0.06), f32(0.9), rim));
  const amb = f32(0.62);
  let lit =
    albedo * (amb + (f32(1) - amb) * diff) * rimAtten +
    vec3f(f32(1), f32(0.98), f32(0.92)) * spec;
  const hN = h / max(ribAmp, f32(1e-4));
  const aoGroove = mix(f32(0.52), f32(1), smoothstep(f32(0), f32(0.22), hN));
  lit *= aoGroove;

  const y = clamp(uv.y, f32(-1), f32(1));
  const indent = f32(0.09);
  const maxY = f32(1) - indent * (f32(1) - hN);
  const edgeDist = maxY - abs(y);
  const fw = max(fwidth(edgeDist), f32(1e-5));
  const profileA = clamp(edgeDist / fw, f32(0), f32(1));
  const outA = a * profileA;
  return vec4f(lit, outA);
});

export const stringShadowFragment = tgpu.fragmentFn({
  in: { uv: vec2f },
  out: vec4f,
})(({ uv }) => {
  "use gpu";
  const a = lineBindLayout.$.uniforms.trailLayerAlpha * f32(0.2);
  const y = clamp(uv.y, f32(-1), f32(1));
  const yAbs = abs(y);
  const falloff = f32(1) - smoothstep(f32(0.15), f32(0.85), yAbs);
  const rgb = vec3f(f32(0.12), f32(0.13), f32(0.16));
  const outA = a * falloff;
  return vec4f(rgb, outA);
});

export const presentFragment = tgpu.fragmentFn({
  in: { uv: vec2f },
  out: vec4f,
})(({ uv }) => {
  "use gpu";
  return textureSample(presentLayout.$.scene, presentLayout.$.samp, uv);
});

export const alphaBlend: GPUBlendState = {
  color: {
    operation: "add",
    srcFactor: "src-alpha",
    dstFactor: "one-minus-src-alpha",
  },
  alpha: {
    operation: "add",
    srcFactor: "one",
    dstFactor: "one-minus-src-alpha",
  },
};

export { caps, endCapSlot, startCapSlot };
