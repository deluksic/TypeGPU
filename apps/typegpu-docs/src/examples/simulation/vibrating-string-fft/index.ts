/**
 * Fixed-end string (DST-I / sine basis) with modal evolution, `@typegpu/sort` `createFft1d` on the
 * odd-extended vector for round-trip sanity check, `@typegpu/geometry` wide lines (control points built
 * in the vertex shader), and chunked Web Audio with gain ducking on re-pluck.
 */
import {
  caps,
  endCapSlot,
  LineControlPoint,
  lineSegmentIndices,
  lineSegmentVariableWidth,
  startCapSlot,
} from '@typegpu/geometry';
import { createFft1d, decomposeWorkgroups } from '@typegpu/sort';
import tgpu, { d } from 'typegpu';
import { fullScreenTriangle } from 'typegpu/common';
import {
  arrayOf,
  builtin,
  f32,
  i32,
  struct,
  u16,
  u32,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data';
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
} from 'typegpu/std';
import { defineControls } from '../../common/defineControls.ts';

// --- DST-I grid: N interior samples, odd extension length L = 2(N+1) (power of two).
const N = 255;
const L = 2 * (N + 1); // 1024
/** GPU synthesize + draw: spatial samples within one frame’s sim interval (no history). */
const INTRA_FRAME_SAMPLES = 20;
const SPATIAL_Y_FLOATS = N * INTRA_FRAME_SAMPLES;
const INTRA_FRAME_LAYER_ALPHA = (20 * 0.1) / INTRA_FRAME_SAMPLES;

/** Apex u clamped to interior nodes j=1..N-2 so the first/last mass points stay fixed. */
const PLUCK_U_MIN = 2 / (N + 1);
const PLUCK_U_MAX = (N - 1) / (N + 1);

const SAMPLE_RATE = 48_000;
const CHUNK_SAMPLES = 512;
const LOOKAHEAD_SEC = 0.22;
/** NDC x at string anchor points; must match `ndcCap` in lineVertex (iso span / projection). */
const STRING_NDC_X_CAP = 1.06;

function embedOddExtension(interior: Float32Array): Float32Array {
  const z = new Float32Array(L);
  z[0] = 0;
  z.set(interior, 1);
  z[N + 1] = 0;
  for (let j = 1; j <= N; j++) {
    z[L - j] = -interior[N - j];
  }
  return z;
}

/** DST-I forward (same sine kernel as inverse up to scale). */
function dst1Forward(interior: Float32Array): Float32Array {
  const out = new Float32Array(N);
  const scale = Math.PI / (N + 1);
  for (let k = 0; k < N; k++) {
    let s = 0;
    for (let j = 0; j < N; j++) {
      s += interior[j] * Math.sin(scale * (j + 1) * (k + 1));
    }
    out[k] = s;
  }
  return out;
}

/** Inverse DST-I: x[j] = (2/(N+1)) * sum_k X[k] sin(...). */
function dst1Inverse(modes: Float32Array): Float32Array {
  const out = new Float32Array(N);
  const c = 2 / (N + 1);
  const scale = Math.PI / (N + 1);
  for (let j = 0; j < N; j++) {
    let s = 0;
    for (let k = 0; k < N; k++) {
      s += modes[k] * Math.sin(scale * (j + 1) * (k + 1));
    }
    out[j] = c * s;
  }
  return out;
}

function buildSinTable(): Float32Array {
  const t = new Float32Array(N * N);
  const scale = Math.PI / (N + 1);
  for (let m = 0; m < N; m++) {
    for (let j = 0; j < N; j++) {
      t[m * N + j] = Math.sin(scale * (j + 1) * (m + 1));
    }
  }
  return t;
}

const root = await tgpu.init({
  adapter: { powerPreference: 'high-performance' },
});
const device = root.device;
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
/** MSAA + stacked trail layers blend here first; 8-bit swapchain would quantize many faint alphas. */
const HDR_COLOR_FORMAT = 'rgba16float';

const fft = createFft1d(root, { n: L, numLines: 2 });

const sinTable = buildSinTable();
const sinTableBuffer = root
  .createBuffer(arrayOf(f32, N * N), [...sinTable])
  .$usage('storage');

const SimUniform = struct({
  n: u32,
  omega0: f32,
  stiffnessB: f32,
  dampingGamma: f32,
  /** Per-mode γ_eff = γ·dampMul; dampMul grows with ω → envelope exp(-γ_eff·t) kills HF faster over time (not a static EQ). */
  spectralDamping: f32,
  /** Audio only: per-mode weight 1/(1+roll·(ω/ω₀)²) softens velocity highs (less metallic). */
  audioVelRolloff: f32,
  simTime: f32,
  synthScale: f32,
  pickupU: f32,
  chunkStartTime: f32,
  invSampleRate: f32,
  chunkSampleCount: u32,
  simTimeFrameStart: f32,
  simTimeFrameEnd: f32,
});

const simUniformBuffer = root
  .createBuffer(SimUniform, {
    n: N,
    // rad/s; fundamental Hz ~ omega0 / (2*pi)
    omega0: 1500,
    stiffnessB: 0.0008,
    dampingGamma: 0.35,
    spectralDamping: 2,
    audioVelRolloff: 0.5,
    simTime: 0,
    synthScale: 2 / (N + 1),
    pickupU: 0.22,
    chunkStartTime: 0,
    invSampleRate: 1 / SAMPLE_RATE,
    chunkSampleCount: CHUNK_SAMPLES,
    simTimeFrameStart: 0,
    simTimeFrameEnd: 0,
  })
  .$usage('uniform');

const RenderUniform = struct({
  aspect: f32,
  yScale: f32,
  trailLayerAlpha: f32,
  n: u32,
});

const renderUniformBuffer = root
  .createBuffer(RenderUniform, {
    aspect: 1,
    yScale: 0.55,
    trailLayerAlpha: 1,
    n: N,
  })
  .$usage('uniform');

/** Added in clip NDC after iso→NDC (negative y = shadow below string). */
const LineVertexDrawUniform = struct({
  ndcShift: vec2f,
  radiusScale: f32,
});

const lineVertexDrawBuffer = root
  .createBuffer(LineVertexDrawUniform, { ndcShift: d.vec2f(0, 0), radiusScale: 1 })
  .$usage('uniform');
const lineShadowVertexDrawBuffer = root
  .createBuffer(LineVertexDrawUniform, { ndcShift: d.vec2f(0, -0.1), radiusScale: 1.8 })
  .$usage('uniform');

const modeY0Buffer = root.createBuffer(arrayOf(f32, N)).$usage('storage');
const modeV0Buffer = root.createBuffer(arrayOf(f32, N)).$usage('storage');
const modeV0Zeros = new Float32Array(N);
const spatialYBuffer = root.createBuffer(arrayOf(f32, SPATIAL_Y_FLOATS)).$usage('storage');
const audioChunkBuffer = root.createBuffer(arrayOf(f32, CHUNK_SAMPLES)).$usage('storage');

const synthesizeLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: SimUniform },
  sinTable: { storage: arrayOf(f32, N * N), access: 'readonly' },
  modeY0: { storage: arrayOf(f32, N), access: 'readonly' },
  modeV0: { storage: arrayOf(f32, N), access: 'readonly' },
  spatialY: { storage: arrayOf(f32, SPATIAL_Y_FLOATS), access: 'mutable' },
});

const audioLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: SimUniform },
  sinTable: { storage: arrayOf(f32, N * N), access: 'readonly' },
  modeY0: { storage: arrayOf(f32, N), access: 'readonly' },
  modeV0: { storage: arrayOf(f32, N), access: 'readonly' },
  audioOut: { storage: arrayOf(f32, CHUNK_SAMPLES), access: 'mutable' },
});

const WG = 64;

const synthesizeMultiKernel = tgpu.computeFn({
  workgroupSize: [WG, 1],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const j = gid.x;
  const k = gid.y;
  const n = synthesizeLayout.$.uniforms.n;
  const nSamp = u32(INTRA_FRAME_SAMPLES);
  if (j >= n || k >= nSamp) {
    return;
  }
  const t0 = synthesizeLayout.$.uniforms.simTimeFrameStart;
  const t1 = synthesizeLayout.$.uniforms.simTimeFrameEnd;
  const denom = max(f32(nSamp) - 1.0, 1.0);
  const t = mix(t0, t1, f32(k) / denom);
  const omega0 = synthesizeLayout.$.uniforms.omega0;
  const B = synthesizeLayout.$.uniforms.stiffnessB;
  const gamma = synthesizeLayout.$.uniforms.dampingGamma;
  const spectral = synthesizeLayout.$.uniforms.spectralDamping;
  const scale = synthesizeLayout.$.uniforms.synthScale;
  const omegaSafe = max(omega0, f32(1.0));
  let sum = f32(0);
  for (let m = u32(0); m < n; m = m + 1) {
    const m1 = f32(m) + 1.0;
    const omega = m1 * omega0 * sqrt(1.0 + B * m1 * m1);
    const safeOmega = max(omega, f32(1e-5));
    const y0 = synthesizeLayout.$.modeY0[m];
    const v0 = synthesizeLayout.$.modeV0[m];
    const relFreq = max(safeOmega / omegaSafe - 1.0, 0.0);
    const dampMul = 1.0 + spectral * relFreq * relFreq / f32(40.0);
    const dec = exp(-gamma * dampMul * t);
    const ym =
      dec * (y0 * cos(safeOmega * t) + (v0 / safeOmega) * sin(safeOmega * t));
    const basis = synthesizeLayout.$.sinTable[m * n + j];
    sum = sum + ym * basis;
  }
  synthesizeLayout.$.spatialY[k * n + j] = sum * scale;
});

/** ∂y/∂t at pickup + optional per-mode HF rolloff (velocity is inherently bright). */
const audioKernel = tgpu.computeFn({
  workgroupSize: [WG],
  in: { gid: builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const i = gid.x;
  const count = audioLayout.$.uniforms.chunkSampleCount;
  if (i >= count) {
    return;
  }
  const n = audioLayout.$.uniforms.n;
  const omega0 = audioLayout.$.uniforms.omega0;
  const B = audioLayout.$.uniforms.stiffnessB;
  const gamma = audioLayout.$.uniforms.dampingGamma;
  const spectral = audioLayout.$.uniforms.spectralDamping;
  const t0 = audioLayout.$.uniforms.chunkStartTime;
  const invSr = audioLayout.$.uniforms.invSampleRate;
  const t = t0 + f32(i) * invSr;
  const scale = audioLayout.$.uniforms.synthScale;
  const uPick = audioLayout.$.uniforms.pickupU;
  const roll = audioLayout.$.uniforms.audioVelRolloff;
  const omegaSafe = max(omega0, f32(1.0));
  let sum = f32(0);
  for (let m = u32(0); m < n; m = m + 1) {
    const m1 = f32(m) + 1.0;
    const omega = m1 * omega0 * sqrt(1.0 + B * m1 * m1);
    const safeOmega = max(omega, f32(1e-5));
    const y0 = audioLayout.$.modeY0[m];
    const v0 = audioLayout.$.modeV0[m];
    const relFreq = max(safeOmega / omegaSafe - 1.0, 0.0);
    const dampMul = 1.0 + spectral * relFreq * relFreq / f32(40.0);
    const alpha = gamma * dampMul;
    const dec = exp(-alpha * t);
    const c = cos(safeOmega * t);
    const s = sin(safeOmega * t);
    const b = v0 / safeOmega;
    const oscill = y0 * c + b * s;
    const ymVel = dec * (-alpha * oscill - y0 * safeOmega * s + b * safeOmega * c);
    const omegaRat = safeOmega / omegaSafe;
    const velWt = 1.0 / (1.0 + roll * omegaRat * omegaRat);
    // oxlint-disable-next-line oxc(approx-constant) -- WGSL π for pickup mode weights
    const w = sin(3.141592653589793 * uPick * m1);
    sum = sum + ymVel * w * scale * velWt;
  }
  audioLayout.$.audioOut[i] =
    sum * f32(2500.0) / (f32(n) * omegaSafe);
});

const synthesizeMultiPipeline = root.createComputePipeline({ compute: synthesizeMultiKernel });
const audioPipeline = root.createComputePipeline({ compute: audioKernel });

const synthesizeBg = root.createBindGroup(synthesizeLayout, {
  uniforms: simUniformBuffer,
  sinTable: sinTableBuffer,
  modeY0: modeY0Buffer,
  modeV0: modeV0Buffer,
  spatialY: spatialYBuffer,
});

const audioBg = root.createBindGroup(audioLayout, {
  uniforms: simUniformBuffer,
  sinTable: sinTableBuffer,
  modeY0: modeY0Buffer,
  modeV0: modeV0Buffer,
  audioOut: audioChunkBuffer,
});

const MAX_JOIN = 4;
const indices = lineSegmentIndices(MAX_JOIN);
const indexBuffer = root.createBuffer(arrayOf(u16, indices.length), indices).$usage('index');

const lineBindLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: RenderUniform },
  lineVertexDraw: { uniform: LineVertexDrawUniform },
  spatialY: { storage: arrayOf(f32, SPATIAL_Y_FLOATS), access: 'readonly' },
});

/** y(u) in time-slab `layer` (offset layer*n in spatialY). */
const stringYAtU = tgpu.fn([f32, u32, u32], f32)((along, n, layer) => {
  'use gpu';
  const buf = lineBindLayout.$.spatialY;
  const base = layer * n;
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
  const alpha = (along - tj) / (tjp - tj);
  return mix(buf[base + j], buf[base + j + u32(1)], alpha);
});

const lineVertex = tgpu.vertexFn({
  in: {
    instanceIndex: builtin.instanceIndex,
    vertexIndex: builtin.vertexIndex,
  },
  out: {
    outPos: builtin.position,
    uv: vec2f,
  },
})(({ instanceIndex, vertexIndex }) => {
  'use gpu';
  const aspect = lineBindLayout.$.uniforms.aspect;
  const yScale = lineBindLayout.$.uniforms.yScale;
  const n = lineBindLayout.$.uniforms.n;
  const hxIso = f32(0.86);
  const ndcCap = f32(1.0);
  const kIsoToNdc = ndcCap / hxIso;
  const segCount = n + u32(1);
  const layer = u32(instanceIndex / segCount);
  const seg = instanceIndex - layer * segCount;
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

  const yA = stringYAtU(uA, n, layer) * yScale;
  const yB = stringYAtU(uB, n, layer) * yScale;
  const yC = stringYAtU(uC, n, layer) * yScale;
  const yD = stringYAtU(uD, n, layer) * yScale;

  const rad = f32(0.012) * lineBindLayout.$.lineVertexDraw.radiusScale;
  const A = LineControlPoint({ position: vec2f(xA, yA), radius: rad });
  const B = LineControlPoint({ position: vec2f(xB, yB), radius: rad });
  const C = LineControlPoint({ position: vec2f(xC, yC), radius: rad });
  const D = LineControlPoint({ position: vec2f(xD, yD), radius: rad });

  const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D, MAX_JOIN);
  const vIso = result.vertexPosition;
  const ndcBase = vec2f(kIsoToNdc * vIso.x, kIsoToNdc * aspect * vIso.y);
  const ndc = ndcBase + lineBindLayout.$.lineVertexDraw.ndcShift;
  const coreVertexIndex = select(0, (vertexIndex - 2) & 3, vertexIndex >= 2);
  const nearSide = select(f32(1), f32(-1), coreVertexIndex === 1 || coreVertexIndex === 2);
  return {
    outPos: vec4f(ndc * result.w, 0, result.w),
    uv: vec2f(
      vIso.x * f32(0.5) + f32(0.5),
      select(nearSide, f32(0), vertexIndex < 2),
    ),
  };
});

const lineFragment = tgpu.fragmentFn({
  in: { uv: vec2f },
  out: vec4f,
})(({ uv }) => {
  'use gpu';
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
  const dhDu = select(f32(0), (-ribAmp * u) / sqrt(max(f32(1e-10), f32(1) - u2)), insideDeriv);
  const hPrime = dhDu / Ruv;

  const theta = clamp(uv.y, f32(-1), f32(1)) * f32(1.495);
  const st = sin(theta);
  const ct = cos(theta);
  const S = f32(1) + h;
  const Pu = vec3f(f32(1), hPrime * st, hPrime * ct);
  const Pv = vec3f(f32(0), S * ct, -S * st);
  let N = normalize(cross(Pu, Pv));
  const V = vec3f(f32(0), f32(0), f32(1));
  N = select(N, neg(N), dot(N, V) < f32(0));

  const L = normalize(vec3f(f32(0.32), f32(0.65), f32(0.68)));
  const diff = max(f32(0), dot(N, L));
  const Rv = reflect(neg(L), N);
  const spec = pow(max(f32(0), dot(Rv, V)), f32(2)) * f32(0.72);

  const grooveCol = vec3f(0.34, 0.3, 0.26);
  const crestCol = vec3f(0.72, 0.65, 0.58);
  const albedo = mix(grooveCol, crestCol, h / max(ribAmp, f32(1e-4)));

  const rim = min(abs(uv.y), f32(1));
  const rimAtten = mix(f32(1), f32(0.9), smoothstep(f32(0.06), f32(0.9), rim));
  const amb = f32(0.62);
  let lit =
    albedo * (amb + (f32(1) - amb) * diff) * rimAtten + vec3f(f32(1), f32(0.98), f32(0.92)) * spec;
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

const stringShadowFragment = tgpu.fragmentFn({
  in: { uv: vec2f },
  out: vec4f,
})(({ uv }) => {
  'use gpu';
  const a = lineBindLayout.$.uniforms.trailLayerAlpha * f32(0.2);

  const y = clamp(uv.y, f32(-1), f32(1));
  const yAbs = abs(y);
  const falloff = f32(1) - smoothstep(f32(0.15), f32(0.85), yAbs);
  const rgb = vec3f(f32(0.12), f32(0.13), f32(0.16));
  const outA = a * falloff;
  return vec4f(rgb, outA);
});

/** Straight RGB + A. Alpha must use “over” (not add-one-one) or stacked layers look depth-blocked. */
const alphaBlend: GPUBlendState = {
  color: {
    operation: 'add',
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
  },
  alpha: {
    operation: 'add',
    srcFactor: 'one',
    dstFactor: 'one-minus-src-alpha',
  },
};

const presentLayout = tgpu.bindGroupLayout({
  scene: { texture: d.texture2d(d.f32) },
  samp: { sampler: 'filtering' },
});

const presentFragment = tgpu.fragmentFn({
  in: { uv: vec2f },
  out: vec4f,
})(({ uv }) => {
  'use gpu';
  return textureSample(presentLayout.$.scene, presentLayout.$.samp, uv);
});

const presentSampler = root['~unstable'].createSampler({
  magFilter: 'linear',
  minFilter: 'linear',
});

const linePipeline = root
  .with(startCapSlot, caps.round)
  .with(endCapSlot, caps.round)
  .createRenderPipeline({
    vertex: lineVertex,
    fragment: lineFragment,
    targets: { format: HDR_COLOR_FORMAT, blend: alphaBlend },
    multisample: { count: 4 },
  })
  .withIndexBuffer(indexBuffer);

const stringShadowPipeline = root
  .with(startCapSlot, caps.round)
  .with(endCapSlot, caps.round)
  .createRenderPipeline({
    vertex: lineVertex,
    fragment: stringShadowFragment,
    targets: { format: HDR_COLOR_FORMAT, blend: alphaBlend },
    multisample: { count: 4 },
  })
  .withIndexBuffer(indexBuffer);

const presentPipeline = root.createRenderPipeline({
  vertex: fullScreenTriangle,
  fragment: presentFragment,
  targets: { format: presentationFormat },
});

const lineBindGroup = root.createBindGroup(lineBindLayout, {
  uniforms: renderUniformBuffer,
  lineVertexDraw: lineVertexDrawBuffer,
  spatialY: spatialYBuffer,
});

const lineShadowBindGroup = root.createBindGroup(lineBindLayout, {
  uniforms: renderUniformBuffer,
  lineVertexDraw: lineShadowVertexDrawBuffer,
  spatialY: spatialYBuffer,
});

spatialYBuffer.write(Array.from(new Float32Array(SPATIAL_Y_FLOATS)));
modeY0Buffer.write(Array.from(new Float32Array(N)));
modeV0Buffer.write(Array.from(modeV0Zeros));

// --- Interaction & audio state
let simOriginMs = performance.now();
/** Sim time at the start of the last presented frame (intra-frame motion streak). */
let prevFrameSimT = 0;
let dragging = false;
let currentU = 0.5;
let pluckHeight = 0.15;
let hasActiveString = false;

const previewInterior = new Float32Array(N);
const previewSpatialFull = new Float32Array(SPATIAL_Y_FLOATS);

function writeDragPreviewSpatialY() {
  for (let k = 0; k < INTRA_FRAME_SAMPLES; k++) {
    previewSpatialFull.set(previewInterior, k * N);
  }
  spatialYBuffer.write(Array.from(previewSpatialFull));
}

/** Linear tent on [uLo,uHi] with peak (uPeak, h); h may be negative (pull down). */
function tentAtPeak(u: number, uLo: number, uPeak: number, uHi: number, h: number): number {
  if (u <= uLo || u >= uHi || uHi - uLo < 1e-8) {
    return 0;
  }
  if (u <= uPeak) {
    const d = uPeak - uLo;
    return d < 1e-8 ? 0 : (h * (u - uLo)) / d;
  }
  const d = uHi - uPeak;
  return d < 1e-8 ? 0 : (h * (uHi - u)) / d;
}

/**
 * Full-string triangle: fixed ends at u=0 and u=1 (y=0), apex at (currentU, pluckHeight).
 */
function pluckDisplacement(u: number): number {
  const h = pluckHeight;
  if (Math.abs(h) < 1e-7) {
    return 0;
  }
  const c = clampScalar(currentU, PLUCK_U_MIN, PLUCK_U_MAX);
  return tentAtPeak(u, 0, c, 1, h);
}

function fillPreviewInterior() {
  for (let j = 0; j < N; j++) {
    const u = (j + 1) / (N + 1);
    previewInterior[j] = pluckDisplacement(u);
  }
}

/** Inverse of ndc_x = (2u-1) * STRING_NDC_X_CAP (isotropic line map). */
function pointerToStringU(clientX: number): number {
  const rect = canvas.getBoundingClientRect();
  const tn = (clientX - rect.left) / Math.max(1e-6, rect.width);
  const ndcX = 2 * tn - 1;
  return clampScalar(0.5 * (1 + ndcX / STRING_NDC_X_CAP), 0, 1);
}

/** Mouse up → positive pluck (string up); mouse down → negative (string down). Uses element rect. */
function pointerToPluckHeight(clientY: number): number {
  const rect = canvas.getBoundingClientRect();
  const tn = (clientY - rect.top) / Math.max(1e-6, rect.height);
  const ndcY = 1 - 2 * tn;
  return clampScalar(ndcY * 0.55, -0.65, 0.65);
}

function pointerToPluckU(clientX: number): number {
  return clampScalar(pointerToStringU(clientX), PLUCK_U_MIN, PLUCK_U_MAX);
}

function clampScalar(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

function updateAspectUniform() {
  const w = Math.max(1, canvas.width);
  const h = Math.max(1, canvas.height);
  renderUniformBuffer.writePartial({ aspect: w / h });
}

let hdrMsaaTex: GPUTexture | undefined;
let hdrMsaaView: GPUTextureView | undefined;
let hdrResolveTexture: ReturnType<(typeof root)['~unstable']['createTexture']> | undefined;
let hdrResolvePassView: GPUTextureView | undefined;
let presentBindGroup: ReturnType<typeof root.createBindGroup> | undefined;
let hdrTargetW = 0;
let hdrTargetH = 0;

function ensureHdrTargets() {
  const w = Math.max(1, canvas.width);
  const h = Math.max(1, canvas.height);
  if (hdrTargetW === w && hdrTargetH === h && hdrResolveTexture && !hdrResolveTexture.destroyed) {
    return;
  }
  hdrTargetW = w;
  hdrTargetH = h;
  hdrMsaaTex?.destroy();
  hdrResolveTexture?.destroy();
  hdrMsaaTex = device.createTexture({
    size: [w, h],
    format: HDR_COLOR_FORMAT,
    sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  hdrMsaaView = hdrMsaaTex.createView();
  hdrResolveTexture = root['~unstable']
    .createTexture({ size: [w, h], format: HDR_COLOR_FORMAT })
    .$usage('sampled', 'render');
  hdrResolvePassView = root.unwrap(hdrResolveTexture).createView();
  presentBindGroup = root.createBindGroup(presentLayout, {
    // oxlint-disable-next-line typescript/no-explicit-any -- createView overload requires $usage('storage') in types only
    scene: (hdrResolveTexture as any).createView(d.texture2d(d.f32)),
    samp: presentSampler,
  });
}

const resizeObserver = new ResizeObserver(() => {
  ensureHdrTargets();
  updateAspectUniform();
});
resizeObserver.observe(canvas);
ensureHdrTargets();
updateAspectUniform();

// --- Web Audio (gain applied when staging chunks; compressor/limiter TODO for more even level)
const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
simUniformBuffer.writePartial({ invSampleRate: 1 / audioCtx.sampleRate });
const masterGain = audioCtx.createGain();
masterGain.gain.value = 1;
masterGain.connect(audioCtx.destination);

/** Linear gain before hard clip into [-1, 1] when building each chunk. */
let audioOutputGain = 1.75;

let sessionGain = audioCtx.createGain();
sessionGain.gain.value = 1;
sessionGain.connect(masterGain);

let nextScheduleTime = 0;
let audioSimCursor = 0;
let audioRunning = false;

/** One pump at a time — overlapping async pumps corrupt cursor/schedule and cause chunk clicks. */
let pumpAudioChain: Promise<void> = Promise.resolve();

function silenceAndNewSession() {
  const now = audioCtx.currentTime;
  sessionGain.gain.cancelScheduledValues(now);
  sessionGain.gain.setValueAtTime(sessionGain.gain.value, now);
  sessionGain.gain.linearRampToValueAtTime(0, now + 0.004);
  const old = sessionGain;
  setTimeout(() => {
    old.disconnect();
  }, 50);
  sessionGain = audioCtx.createGain();
  sessionGain.gain.value = 0;
  sessionGain.connect(masterGain);
  nextScheduleTime = now + 0.006;
  audioRunning = false;
  audioSimCursor = 0;
}

function startSessionGainAt(t0: number) {
  sessionGain.gain.cancelScheduledValues(t0);
  sessionGain.gain.setValueAtTime(0, t0);
  sessionGain.gain.linearRampToValueAtTime(1, t0 + 0.02);
}

async function runEmbedFftRoundTrip(interior: Float32Array) {
  const z = embedOddExtension(interior);
  const packed: d.v2f[] = [];
  for (let i = 0; i < L; i++) {
    packed.push(d.vec2f(z[i], 0));
  }
  for (let i = 0; i < L; i++) {
    packed.push(d.vec2f(0, 0));
  }
  fft.input.write(packed);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  fft.encodeForward(pass);
  fft.encodeInverse(pass);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const out = (await fft.output().read()) as { x: number; y: number }[];
  let maxErr = 0;
  for (let i = 0; i < L; i++) {
    maxErr = Math.max(maxErr, Math.abs(out[i].x - z[i]));
  }
  console.info('[vibrating-string-fft] Odd-embed GPU FFT round-trip max |Δz|:', maxErr);
}

function pumpAudio() {
  pumpAudioChain = pumpAudioChain
    .catch(() => {})
    .then(() => pumpAudioOnce());
}

async function pumpAudioOnce() {
  if (!hasActiveString || !audioRunning) {
    return;
  }
  let guard = 0;
  while (guard < 8) {
    guard++;
    const now = audioCtx.currentTime;
    if (nextScheduleTime >= now + LOOKAHEAD_SEC) {
      break;
    }
    simUniformBuffer.writePartial({
      chunkStartTime: audioSimCursor,
      chunkSampleCount: CHUNK_SAMPLES,
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    audioPipeline.with(audioBg).dispatchWorkgroups(...decomposeWorkgroups(Math.ceil(CHUNK_SAMPLES / WG)));
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const raw = await audioChunkBuffer.read();
    const arr = raw as unknown as number[] | Float32Array;
    const tmp = new Float32Array(CHUNK_SAMPLES);
    if (arr instanceof Float32Array) {
      tmp.set(arr.subarray(0, CHUNK_SAMPLES));
    } else {
      for (let i = 0; i < CHUNK_SAMPLES; i++) {
        tmp[i] = arr[i] ?? 0;
      }
    }
    const g = audioOutputGain;
    for (let i = 0; i < CHUNK_SAMPLES; i++) {
      const x = tmp[i] * g;
      tmp[i] = x <= -1 ? -1 : x >= 1 ? 1 : x;
    }
    const sr = audioCtx.sampleRate;
    const buf = audioCtx.createBuffer(1, CHUNK_SAMPLES, sr);
    buf.copyToChannel(tmp, 0, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(sessionGain);
    const tClock = audioCtx.currentTime;
    const t0 = Math.max(nextScheduleTime, tClock + 0.002);
    src.start(t0);
    nextScheduleTime = t0 + CHUNK_SAMPLES / sr;
    audioSimCursor += CHUNK_SAMPLES / sr;
  }
}

function commitPluck() {
  fillPreviewInterior();
  const modes = dst1Forward(previewInterior);
  modeY0Buffer.write(Array.from(modes));
  modeV0Buffer.write(Array.from(modeV0Zeros));
  const cpuSpatial = dst1Inverse(modes);
  let err = 0;
  for (let j = 0; j < N; j++) {
    err = Math.max(err, Math.abs(cpuSpatial[j] - previewInterior[j]));
  }
  console.info('[vibrating-string-fft] DST round-trip max |Δy| (CPU):', err);

  simOriginMs = performance.now();
  prevFrameSimT = 0;
  hasActiveString = true;
  audioRunning = true;
  const acNow = audioCtx.currentTime;
  audioSimCursor = 0;
  nextScheduleTime = acNow + 0.02;
  startSessionGainAt(acNow);
  void runEmbedFftRoundTrip(previewInterior);
}

let activePluckPointerId: number | undefined;

function releasePluckPointer(e: PointerEvent) {
  if (activePluckPointerId === e.pointerId) {
    canvas.releasePointerCapture(e.pointerId);
    activePluckPointerId = undefined;
  }
}

canvas.addEventListener('pointerdown', (e) => {
  void audioCtx.resume();
  silenceAndNewSession();
  canvas.setPointerCapture(e.pointerId);
  activePluckPointerId = e.pointerId;
  dragging = true;
  currentU = pointerToPluckU(e.clientX);
  pluckHeight = pointerToPluckHeight(e.clientY);
  fillPreviewInterior();
  writeDragPreviewSpatialY();
});

canvas.addEventListener('pointermove', (e) => {
  if (!dragging) {
    return;
  }
  currentU = pointerToPluckU(e.clientX);
  pluckHeight = pointerToPluckHeight(e.clientY);
  fillPreviewInterior();
  writeDragPreviewSpatialY();
});

canvas.addEventListener('pointerup', (e) => {
  if (!dragging) {
    return;
  }
  dragging = false;
  currentU = pointerToPluckU(e.clientX);
  pluckHeight = pointerToPluckHeight(e.clientY);
  commitPluck();
  releasePluckPointer(e);
});

canvas.addEventListener('pointercancel', (e) => {
  if (!dragging) {
    return;
  }
  dragging = false;
  currentU = pointerToPluckU(e.clientX);
  pluckHeight = pointerToPluckHeight(e.clientY);
  commitPluck();
  releasePluckPointer(e);
});

/** e.g. browser UI stealing focus — pointerup may not fire */
canvas.addEventListener('lostpointercapture', () => {
  activePluckPointerId = undefined;
  if (dragging) {
    dragging = false;
    commitPluck();
  }
});

let frameId = 0;

const draw = () => {
  ensureHdrTargets();
  const msaa = hdrMsaaView;
  const bg = presentBindGroup;
  const resolveView = hdrResolvePassView;
  if (!msaa || !bg || !resolveView) {
    return;
  }
  const simT = (performance.now() - simOriginMs) / 1000;
  const tStart = prevFrameSimT;
  let tEnd = simT;
  if (tEnd <= tStart) {
    tEnd = tStart + 1e-5;
  }
  simUniformBuffer.writePartial({
    simTime: simT,
    simTimeFrameStart: tStart,
    simTimeFrameEnd: tEnd,
  });

  if (!dragging) {
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    synthesizeMultiPipeline
      .with(synthesizeBg)
      .dispatchWorkgroups(Math.ceil(N / WG), INTRA_FRAME_SAMPLES, 1);
    cp.end();
    device.queue.submit([enc.finish()]);
  }

  renderUniformBuffer.writePartial({ trailLayerAlpha: INTRA_FRAME_LAYER_ALPHA });
  const instanceCount = (N + 1) * INTRA_FRAME_SAMPLES;

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: msaa,
        resolveTarget: resolveView,
        clearValue: [0.97, 0.97, 0.98, 1],
        loadOp: 'clear',
        storeOp: 'discard',
      },
    ],
  });

  stringShadowPipeline
    .with(lineShadowBindGroup)
    .with(pass)
    .drawIndexed(indices.length, instanceCount);

  linePipeline.with(lineBindGroup).with(pass).drawIndexed(indices.length, instanceCount);

  pass.end();

  const passPresent = enc.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  presentPipeline.with(bg).with(passPresent).draw(3);
  passPresent.end();

  device.queue.submit([enc.finish()]);

  prevFrameSimT = simT;

  pumpAudio();
};

const loop = () => {
  draw();
  frameId = requestAnimationFrame(loop);
};
frameId = requestAnimationFrame(loop);

export const controls = defineControls({
  'Fundamental ω₀ (rad/s)': {
    initial: 1500,
    min: 1,
    max: 8000,
    step: 50,
    onSliderChange: (v) => {
      simUniformBuffer.writePartial({ omega0: v });
    },
  },
  Stiffness: {
    initial: 0.0008,
    min: 0,
    max: 0.01,
    step: 0.0001,
    onSliderChange: (v) => {
      simUniformBuffer.writePartial({ stiffnessB: v });
    },
  },
  Damping: {
    initial: 0.35,
    min: 0.05,
    max: 2,
    step: 0.05,
    onSliderChange: (v) => {
      simUniformBuffer.writePartial({ dampingGamma: v });
    },
  },
  'HF decay (spectral)': {
    initial: 2,
    min: 0,
    max: 20,
    step: 0.5,
    onSliderChange: (v) => {
      simUniformBuffer.writePartial({ spectralDamping: v });
    },
  },
  'Warmth (audio)': {
    initial: 0,
    min: 0,
    max: 1.5,
    step: 0.05,
    onSliderChange: (v) => {
      simUniformBuffer.writePartial({ audioVelRolloff: v });
    },
  },
  'Pickup u': {
    initial: 0.22,
    min: 0.05,
    max: 0.95,
    step: 0.01,
    onSliderChange: (v) => {
      simUniformBuffer.writePartial({ pickupU: v });
    },
  },
  'Visual Y scale': {
    initial: 0.55,
    min: 0.15,
    max: 1.2,
    step: 0.01,
    onSliderChange: (v) => {
      renderUniformBuffer.writePartial({ yScale: v });
    },
  },
  'Audio gain': {
    initial: 0.25,
    min: 0.25,
    max: 6,
    step: 0.25,
    onSliderChange: (v) => {
      audioOutputGain = v;
    },
  },
});

export function onCleanup() {
  cancelAnimationFrame(frameId);
  resizeObserver.disconnect();
  hdrMsaaTex?.destroy();
  hdrResolveTexture?.destroy();
  fft.destroy();
  root.destroy();
  root.device.destroy();
}