/** DST-I grid: N interior samples, odd extension length L = 2(N+1). */
export const N = 255;
export const L = 2 * (N + 1); // 512

/** Number of strings (bass guitar). */
export const S = 8;

/** Spatial samples within one frame's sim interval (motion blur layers). */
export const INTRA_FRAME_SAMPLES = 1;
export const SPATIAL_Y_PER_STRING = N * INTRA_FRAME_SAMPLES;
export const SPATIAL_Y_FLOATS = S * SPATIAL_Y_PER_STRING;
export const INTRA_FRAME_LAYER_ALPHA = (20 * 0.1) / INTRA_FRAME_SAMPLES;

export const PLUCK_U_MIN = 2 / (N + 1);
export const PLUCK_U_MAX = (N - 1) / (N + 1);

export const SAMPLE_RATE = 48_000;
export const CHUNK_SAMPLES = 512;
export const LOOKAHEAD_SEC = 0.22;

/** NDC x at string anchor points; must match ndcCap in lineVertex. */
export const STRING_NDC_X_CAP = 1.06;

export const WG = 64;
export const AUDIO_WG = 256;
export const AUDIO_SAMPLES_PER_WG = Math.floor(AUDIO_WG / S);
export const MAX_JOIN = 4;
export const HDR_COLOR_FORMAT = "rgba16float" as const;

export const SYNTH_SCALE = 2 / (N + 1);

export const DAMPER_U_START = 0.8;
export const DAMPER_U_END = 0.98;
export const DAMPER_STRENGTH = 15;

export function clampScalar(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

export function buildSinTable(): Float32Array {
  const t = new Float32Array(N * N);
  const scale = Math.PI / (N + 1);
  for (let m = 0; m < N; m++) {
    for (let j = 0; j < N; j++) {
      t[m * N + j] = Math.sin(scale * (j + 1) * (m + 1));
    }
  }
  return t;
}
