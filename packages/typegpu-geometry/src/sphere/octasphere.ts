import { type v3f } from 'typegpu/data';
import { segmentSphericalTriangleSlot } from './segmentSphericalTriangle/slots.ts';
import {
  OCTAHEDRON_FACE_COUNT,
  octahedronFaceVertices,
} from '../internal/octahedronFace.ts';
import { makePatchInstancingHelpers } from '../internal/patchCountHelpers.ts';
import { ProceduralShapeResult } from '../shape/result.ts';

export { OCTAHEDRON_FACE_COUNT };

const instancing = makePatchInstancingHelpers(OCTAHEDRON_FACE_COUNT);

export const octasphereIndexCountPerFace = instancing.indexCountPerPatch;
export const octasphereIndexCount = instancing.indexCount;
export const octasphereWireframeIndexCountPerFace = instancing.wireframeIndexCountPerPatch;
export const octasphereWireframeIndexCount = instancing.wireframeIndexCount;
export const octasphereInstanceCount = instancing.instanceCount;
export const octasphereObjectIndex = instancing.objectIndex;
export const octaspherePatchIndex = instancing.patchIndex;

/**
 * Vertex shader helper for procedural octaspheres.
 *
 * Setup: one instance per octahedron face (`octasphereInstanceCount(n)`), shared
 * `segmentTriangleIndices(maxSegmentCount)` index buffer, draw a prefix via `octasphereIndexCountPerFace(segmentCount)`.
 * Pass `@builtin(instance_index)` / `@builtin(vertex_index)`; `segmentCount` must match the draw count.
 * Scale `vertex` by radius and offset by object center for world position.
 */
export function octasphere(instanceIndex: number, vertexIndex: number, segmentCount: number) {
  'use gpu';
  const faceIndex = octaspherePatchIndex(instanceIndex);
  const faceOffset = faceIndex * 3;
  const a = octahedronFaceVertices.$[faceOffset] as v3f;
  const b = octahedronFaceVertices.$[faceOffset + 1] as v3f;
  const c = octahedronFaceVertices.$[faceOffset + 2] as v3f;
  const vertex = segmentSphericalTriangleSlot.$(a, b, c, vertexIndex, segmentCount);
  return ProceduralShapeResult({
    vertex,
    normal: vertex,
  });
}
