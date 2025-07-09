import tgpu from 'typegpu';
import { bool, f32, struct, vec2f } from 'typegpu/data';
import {
  abs,
  add,
  dot,
  length,
  mul,
  normalize,
  select,
  sqrt,
  sub,
} from 'typegpu/std';

export const ortho2d = tgpu.fn([vec2f], vec2f)((v) => {
  return vec2f(-v.y, v.x);
});

export const ortho2dNeg = tgpu.fn([vec2f], vec2f)((v) => {
  return vec2f(v.y, -v.x);
});

export const cross2d = tgpu.fn([vec2f, vec2f], f32)((a, b) => {
  return a.x * b.y - a.y * b.x;
});

/**
 * Finds mid direction between two vectors. The direction will always be on the CW arc
 * between the vectors.
 */
export const midDirection = tgpu.fn([vec2f, vec2f], vec2f)((a, b) => {
  const cos = dot(a, b);
  const sin = cross2d(a, b);
  // we avoid flicker by allowing almost-colinear vectors be treated as such.
  const sinSign = select(f32(-1), f32(1), sin < -1e-6);
  const orthoA = ortho2dNeg(a);
  const orthoB = ortho2d(b);
  const dir = select(mul(add(a, b), sinSign), add(orthoA, orthoB), cos < 0);
  return normalize(dir);
});

/**
 * Finds mid direction between two vectors. There is no check done to be on the CW part.
 */
export const midDirectionNoCheck = tgpu.fn([vec2f, vec2f], vec2f)((a, b) => {
  return normalize(add(a, b));
});

/**
 * Finds miter point given two vectors on a unit circle. Order of arguments important.
 */
export const miterPoint = tgpu.fn([vec2f, vec2f], vec2f)((a, b) => {
  if (cross2d(a, b) < 0) {
    // if the miter is at infinity, just make it super far
    return mul(normalize(add(a, b)), -1e6);
  }
  const ab = add(a, b);
  const len2 = dot(ab, ab);
  return mul(ab, 2 / len2);
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
    return select(a, b, (dotA >= dotB) === invert);
  },
);

const LimitAlongResult = struct({
  a: vec2f,
  b: vec2f,
});

/**
 * Leaves a and b separate if no collision, otherwise merges them towards "middle".
 */
export const limitTowardsMiddle = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f],
  LimitAlongResult,
)(
  (dir, middle, a, b) => {
    const aX = dot(a, dir);
    const bX = dot(b, dir);
    if (aX >= bX) {
      // a is in front of b, don't touch them
      return LimitAlongResult({ a, b });
    }
    const middleX = dot(middle, dir);
    const same = select(a, b, abs(aX - middleX) > abs(bX - middleX));
    return LimitAlongResult({ a: same, b: same });
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
