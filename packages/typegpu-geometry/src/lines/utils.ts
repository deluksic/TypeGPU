import tgpu from 'typegpu';
import { bool, f32, struct, vec2f } from 'typegpu/data';
import {
  add,
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
import {
  addMul,
  bisectCcw,
  cross2d,
  midPoint,
  rot90ccw,
  rot90cw,
} from '../utils.ts';

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

export const inscribedCenter = tgpu.fn([vec2f, vec2f, vec2f], vec2f)(
  (a, b, c) => {
    const ab = sub(b, a);
    const ac = sub(c, a);
    const bc = sub(c, b);
    const lenAB = length(ab);
    const lenAC = length(ac);
    const lenBC = length(bc);
    return mul(
      add(
        add(
          mul(a, lenBC),
          mul(b, lenAC),
        ),
        mul(c, lenAB),
      ),
      1 / (lenAB + lenAC + lenBC),
    );
  },
);

export const quadCentroid = tgpu.fn([vec2f, vec2f, vec2f, vec2f], vec2f)(
  (a, b, c, d) => {
    const cross0 = cross2d(a, b);
    const cross1 = cross2d(b, c);
    const cross2 = cross2d(c, d);
    const cross3 = cross2d(d, a);

    const area = 0.5 * (cross0 + cross1 + cross2 + cross3);
    const factor = (1.0 / 6.0) / area;

    let sum = mul(add(a, b), cross0);
    sum = add(sum, mul(add(b, c), cross1));
    sum = add(sum, mul(add(c, d), cross2));
    sum = add(sum, mul(add(d, a), cross3));

    return mul(sum, factor);
  },
);
