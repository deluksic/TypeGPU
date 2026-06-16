import { type v3f, vec3f } from 'typegpu/data';
import { sqrt } from 'typegpu/std';

/** Map a cube-surface point to a unit sphere with more even spacing than normalize. */
export function spherify(p: v3f): v3f {
  'use gpu';
  const x2 = p.x * p.x;
  const y2 = p.y * p.y;
  const z2 = p.z * p.z;
  return vec3f(
    p.x * sqrt(1 - 0.5 * (y2 + z2) + (y2 * z2) / 3),
    p.y * sqrt(1 - 0.5 * (z2 + x2) + (z2 * x2) / 3),
    p.z * sqrt(1 - 0.5 * (x2 + y2) + (x2 * y2) / 3),
  );
}
