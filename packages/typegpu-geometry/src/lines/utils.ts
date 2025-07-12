import tgpu from 'typegpu';
import { bool, f32, struct, vec2f } from 'typegpu/data';
import { dot, length, normalize, select, sqrt, sub } from 'typegpu/std';
import { addMul } from '../utils.ts';

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
  (A0, A1, B0, B1) => {
    const int = intersectLines(A0, A1, B0, B1);
    if (int.t <= 0 || int.t >= 1) {
      // they don't intersect, return untouched
      return LimitAlongResult({ a: A1, b: B1 });
    }
    return LimitAlongResult({ a: int.point, b: int.point });
  },
);
