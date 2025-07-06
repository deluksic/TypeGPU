import tgpu from 'typegpu';
import { bool, f32, struct, vec2f } from 'typegpu/data';
import {
  add,
  dot,
  length,
  min,
  mul,
  normalize,
  select,
  sqrt,
  sub,
} from 'typegpu/std';

/**
 * Finds miter point given two vectors on a unit circle.
 */
export const miterPoint = tgpu.fn([vec2f, vec2f], vec2f)((a, b) => {
  const ab = add(a, b);
  const len2 = dot(ab, ab);
  return mul(ab, 2 / len2);
});

export const cross2d = tgpu.fn([vec2f, vec2f], f32)((a, b) => {
  return a.x * b.y - a.y * b.x;
});

export const triangleWinding = tgpu.fn(
  [vec2f, vec2f, vec2f],
  bool,
)(
  (a, b, c) => {
    return cross2d(sub(b, a), sub(c, a)) > 0;
  },
);

export const addMul = tgpu.fn(
  [vec2f, vec2f, f32],
  vec2f,
)(
  (a, b, f) => {
    return add(a, mul(b, f));
  },
);

export const ortho2d = tgpu.fn([vec2f], vec2f)((v) => {
  return vec2f(-v.y, v.x);
});

const clampLength2 = tgpu.fn([vec2f, f32], vec2f)(
  (v, maxLength) => {
    const len = length(v);
    const clampedLen = min(len, maxLength) / select(1, len, len > 0);
    return mul(v, clampedLen);
  },
);

const ExternalNormals = struct({
  n1: vec2f,
  n2: vec2f,
});

/**
 * Computes external tangent directions (normals to tangent)
 * for two circles at a `distance` with radii `r1` and `r2`.
 */
export const externalNormals = tgpu.fn(
  [vec2f, f32, f32],
  ExternalNormals,
)((distance, r1, r2) => {
  const dNorm = normalize(distance);
  const expCos = (r1 - r2) / length(distance);
  const expSin = sqrt(1 - expCos * expCos);
  const t1 = vec2f(
    dNorm.x * expCos - dNorm.y * expSin,
    dNorm.x * expSin + dNorm.y * expCos,
  );
  const t2 = vec2f(
    dNorm.x * expCos + dNorm.y * expSin,
    -dNorm.x * expSin + dNorm.y * expCos,
  );
  return ExternalNormals({ n1: t1, n2: t2 });
});

/**
 * Selects either a or b, depending which one is earlier along dir
 */
export const limitAlong = tgpu.fn([vec2f, vec2f, vec2f, bool], vec2f)(
  (a, b, dir, invert) => {
    const dotA = dot(a, dir);
    const dotB = dot(b, dir);
    return select(a, b, (dotA < dotB) !== invert);
  },
);

const Intersection = struct({
  valid: bool,
  t: f32,
  point: vec2f,
});

export const intersectLines = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f],
  Intersection,
)(
  (a, b, c, d) => {
    const r = sub(b, a);
    const s = sub(d, c);
    const rxs = r.x * s.y - r.y * s.x;
    const AC = sub(c, a);
    const t = (AC.x * s.y - AC.y * s.x) / rxs;
    return {
      valid: rxs !== 0,
      t,
      point: addMul(a, r, t),
    };
  },
);
