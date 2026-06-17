import { vec3f, type v3f } from 'typegpu/data';
import { dot, normalize } from 'typegpu/std';
import { slerpApproxWarp } from '../../utils.ts';
import { triangleGridBarycentrics } from '../../segmentedTriangle.ts';

/** Equalize barycentric weights for spherical edges, then normalize onto the sphere. */
export function uniformArea(
  A: v3f,
  B: v3f,
  C: v3f,
  vertexIndex: number,
  maxSegmentCount: number,
): v3f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, maxSegmentCount);
  const oppositeDots = vec3f(dot(B, C), dot(A, C), dot(A, B));
  const wUniform = vec3f(
    slerpApproxWarp(oppositeDots.x, w.x),
    slerpApproxWarp(oppositeDots.y, w.y),
    slerpApproxWarp(oppositeDots.z, w.z),
  );
  return normalize(A * wUniform.x + B * wUniform.y + C * wUniform.z);
}
