import { f32, type v3f, vec3f } from 'typegpu/data';
import { dot, select } from 'typegpu/std';
import { triangleGridBarycentrics } from '../segmentedTriangle.ts';
import { slerpApprox3 } from '../utils.ts';

/** arcParam at corners A, B, C for the first triangle of a swept-arc quad patch. */
export const ARC_PATCH_PARAMS_TRI0 = vec3f(0, 1, 1);

/** arcParam at corners A, B, C for the second triangle of a swept-arc quad patch. */
export const ARC_PATCH_PARAMS_TRI1 = vec3f(0, 1, 0);

export function arcPatchHeights(triInEdge: number, hMin: number, hMax: number): v3f {
  'use gpu';
  return select(
    vec3f(hMin, hMax, hMax),
    vec3f(hMin, hMin, hMax),
    triInEdge === 0,
  );
}

/**
 * Swept-arc developable patch: blend arcParam and height from corner values, then sweep
 * `slerpApprox3(arcStartDir, arcEndDir, arcParam)` along `axisDir`.
 */
export function arcPatchVertex(
  vertexIndex: number,
  segmentCount: number,
  center: v3f,
  axisDir: v3f,
  arcStartDir: v3f,
  arcEndDir: v3f,
  arcParams: v3f,
  heights: v3f,
  radius: number,
): v3f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, segmentCount);
  const arcParam = dot(w, arcParams);
  const height = dot(w, heights);
  const radial = slerpApprox3(arcStartDir, arcEndDir, arcParam);
  return center + axisDir * height + radial * f32(radius);
}
