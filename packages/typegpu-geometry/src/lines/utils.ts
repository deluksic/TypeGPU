import tgpu from 'typegpu';
import { bool, f32, struct, vec2f } from 'typegpu/data';
import {
  add,
  clamp,
  distance,
  dot,
  length,
  max,
  mix,
  mul,
  normalize,
  sqrt,
  sub,
} from 'typegpu/std';
import {
  addMul,
  bisectCcw,
  cross2d,
  midPoint,
  rot90ccw,
  rot90cw,
} from '../utils.ts';

/**
 * Finds the miter point of tangents to two points on a circle.
 * The miter point is on the smaller arc.
 */
export const miterPointNoCheck = tgpu.fn([vec2f, vec2f], vec2f)((a, b) => {
  const ab = add(a, b);
  return mul(ab, 2 / dot(ab, ab));
});

/**
 * Finds the miter point of tangents to two points on respective circles.
 * The miter point is on the counter-clockwise arc between the circles if possible,
 * otherwise at "infinity".
 */
export const miterPoint = tgpu.fn([vec2f, vec2f], vec2f)((a, b) => {
  const sin_ = cross2d(a, b);
  const bisection = bisectCcw(a, b);
  if (sin_ < 0) {
    // if the miter is at infinity, just make it super far
    return mul(bisection, -1e6);
  }
  const b2 = dot(b, b);
  const cos_ = dot(a, b);
  const diff = b2 - cos_;
  if (diff * diff < 1e-4) {
    // the vectors are almost colinear
    return bisection;
  }
  const t = diff / sin_;
  return addMul(a, rot90ccw(a), t);
});

const MiterLimitResult = struct({
  left: vec2f,
  mid: vec2f,
  right: vec2f,
});

export const miterLimit = tgpu.fn([vec2f, vec2f], MiterLimitResult)(
  (a, b) => {
    const sin_ = cross2d(a, b);
    const bisection = bisectCcw(a, b);
    if (sin_ < 0) {
      // if the miter is at infinity, just make it super far
      const same = mul(bisection, -1e6);
      return {
        left: same,
        mid: same,
        right: same,
      };
    }
    const b2 = dot(b, b);
    const cos_ = dot(a, b);
    const diff = b2 - cos_;
    if (diff * diff < 1e-4) {
      // the vectors are almost colinear
      const same = bisection;
      return {
        left: same,
        mid: same,
        right: same,
      };
    }
    const t = clamp(diff / sin_, 0, 1);
    const left = addMul(a, rot90ccw(a), t);
    const right = addMul(b, rot90cw(b), t);
    return {
      left: left,
      mid: midPoint(left, right),
      right: right,
    };
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
  const expSin = sqrt(max(0, 1 - expCos * expCos));
  const a = dNorm.x * expCos;
  const b = dNorm.y * expSin;
  const c = dNorm.x * expSin;
  const d = dNorm.y * expCos;
  const n1 = vec2f(a - b, c + d);
  const n2 = vec2f(a + b, -c + d);
  return ExternalNormals({ n1, n2 });
});

const Intersection = struct({
  valid: bool,
  t: f32,
  point: vec2f,
});

export const intersectLines = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f],
  Intersection,
)(
  (A1, A2, B1, B2) => {
    const a = sub(A2, A1);
    const b = sub(B2, B1);
    const axb = cross2d(a, b);
    const AB = sub(B1, A1);
    const t = cross2d(AB, b) / axb;
    return {
      valid: axb !== 0,
      t,
      point: addMul(A1, a, t),
    };
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
  (middle, dir, p1, p2) => {
    const t1 = dot(sub(p1, middle), dir);
    const t2 = dot(sub(p2, middle), dir);
    if (t1 < t2) {
      return LimitAlongResult({ a: p1, b: p2 });
    }
    const t = clamp((0 - t1) / (t2 - t1), 0, 1);
    const p = mix(p1, p2, t);
    return LimitAlongResult({ a: p, b: p });
  },
);

export const distanceToLineSegment = tgpu.fn([vec2f, vec2f, vec2f], f32)(
  (A, B, point) => {
    const p = sub(point, A);
    const AB = sub(B, A);
    const t = clamp(dot(p, AB) / dot(AB, AB), 0, 1);
    const projP = addMul(A, AB, t);
    return distance(point, projP);
  },
);
