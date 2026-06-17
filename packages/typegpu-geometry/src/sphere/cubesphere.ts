import { type v3f } from 'typegpu/data';
import { normalize } from 'typegpu/std';
import { CUBE_FACE_TRIANGLE_COUNT, cubeFaceTriangles } from '../cubeFaceTriangles.ts';
import { segmentTriangle3 } from '../segmentedTriangle.ts';
import { makePatchInstancingHelpers } from '../internal/patchCountHelpers.ts';
import { ProceduralShapeResult } from '../shape/result.ts';
import { spherify } from './spherify.ts';

export { CUBE_FACE_TRIANGLE_COUNT };

const instancing = makePatchInstancingHelpers(CUBE_FACE_TRIANGLE_COUNT);

export const cubesphereIndexCountPerFace = instancing.indexCountPerPatch;
export const cubesphereIndexCount = instancing.indexCount;
export const cubesphereWireframeIndexCountPerFace = instancing.wireframeIndexCountPerPatch;
export const cubesphereWireframeIndexCount = instancing.wireframeIndexCount;
export const cubesphereInstanceCount = instancing.instanceCount;
export const cubesphereObjectIndex = instancing.objectIndex;
export const cubespherePatchIndex = instancing.patchIndex;

/**
 * Vertex shader helper for procedural cubespheres.
 *
 * Each cube face is two triangles (BL–BR–TR and BL–TR–TL). One instance per
 * face triangle (`cubesphereInstanceCount(n)`), shared `segmentTriangleIndices(maxSegmentCount)`
 * index buffer, draw a prefix via `cubesphereIndexCountPerFace(segmentCount)`.
 * Vertices are linearly interpolated on the cube, then mapped with `spherify`.
 */
export function cubesphere(instanceIndex: number, vertexIndex: number, segmentCount: number) {
  'use gpu';
  const faceTriangleIndex = cubespherePatchIndex(instanceIndex);
  const cornerOffset = faceTriangleIndex * 3;
  const a = cubeFaceTriangles.$[cornerOffset] as v3f;
  const b = cubeFaceTriangles.$[cornerOffset + 1] as v3f;
  const c = cubeFaceTriangles.$[cornerOffset + 2] as v3f;
  const onCube = segmentTriangle3(a, b, c, vertexIndex, segmentCount);
  const vertex = normalize(spherify(onCube));
  return ProceduralShapeResult({
    vertex,
    normal: vertex,
  });
}
