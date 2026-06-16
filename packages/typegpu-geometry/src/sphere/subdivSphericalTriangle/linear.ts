import { type v3f } from 'typegpu/data';
import { normalize } from 'typegpu/std';
import { triangleGridBarycentrics } from '../../subdividedTriangle.ts';

/** Linear barycentrics projected onto the unit sphere (chordal). */
export function linear(A: v3f, B: v3f, C: v3f, vertexIndex: number, maxSubdivCount: number): v3f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, maxSubdivCount);
  return normalize(A * w.x + B * w.y + C * w.z);
}
