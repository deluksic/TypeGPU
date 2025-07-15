import tgpu from 'typegpu';
import { bool, f32, struct, vec2f } from 'typegpu/data';
import {
  clamp,
  distance,
  dot,
  length,
  mix,
  mul,
  normalize,
  select,
  sqrt,
  sub,
} from 'typegpu/std';
import { addMul, bisectCcw, cross2d, rot90ccw } from '../utils.ts';

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
  (M, dir, p1, p2) => {
    const t1 = dot(sub(p1, M), dir);
    const t2 = dot(sub(p2, M), dir);
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
