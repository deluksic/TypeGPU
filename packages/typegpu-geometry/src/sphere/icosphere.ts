import tgpu from 'typegpu';
import { arrayOf, u32, type v3f, vec3f } from 'typegpu/data';
import { normalize } from 'typegpu/std';
import { subdivSphericalTriangleSlot } from './subdivSphericalTriangle/slots.ts';
import {
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../subdividedTriangle.ts';
import { ProceduralSphereResult } from './result.ts';

export const ICOSAHEDRON_FACE_COUNT = 20;

const goldenRatio = (1 + Math.sqrt(5)) / 2;

const icosahedronVerticesData = (
  [
    [-1, goldenRatio, 0],
    [1, goldenRatio, 0],
    [-1, -goldenRatio, 0],
    [1, -goldenRatio, 0],
    [0, -1, goldenRatio],
    [0, 1, goldenRatio],
    [0, -1, -goldenRatio],
    [0, 1, -goldenRatio],
    [goldenRatio, 0, -1],
    [goldenRatio, 0, 1],
    [-goldenRatio, 0, -1],
    [-goldenRatio, 0, 1],
  ] as const
).map(([x, y, z]) => normalize(vec3f(x, y, z)));

const icosahedronFaceCornerIndices = [
  0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
  3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
] as const;

const icosahedronFaceVerticesData = icosahedronFaceCornerIndices.map(
  (index) => icosahedronVerticesData[index],
) as v3f[];

const icosahedronFaceVertices = tgpu.const(
  arrayOf(vec3f, icosahedronFaceVerticesData.length),
  icosahedronFaceVerticesData,
);

export function icosphereIndexCountPerFace(subdivisions: number) {
  return subdivTriangleIndexCount(subdivisions);
}

export function icosphereIndexCount(subdivisions: number) {
  return ICOSAHEDRON_FACE_COUNT * icosphereIndexCountPerFace(subdivisions);
}

export function icosphereWireframeIndexCountPerFace(subdivisions: number) {
  return subdivTriangleWireframeIndexCount(subdivisions);
}

export function icosphereWireframeIndexCount(subdivisions: number) {
  return ICOSAHEDRON_FACE_COUNT * icosphereWireframeIndexCountPerFace(subdivisions);
}

export function icosphereInstanceCount(objectCount: number) {
  return objectCount * ICOSAHEDRON_FACE_COUNT;
}

/**
 * Vertex shader helper for procedural icospheres.
 *
 * Setup: one instance per icosahedron face (`icosphereInstanceCount(n)`), shared
 * `subdivTriangleIndices(maxSubdiv)` index buffer, draw a prefix via `icosphereIndexCountPerFace(subdivCount)`.
 * Pass `@builtin(instance_index)` / `@builtin(vertex_index)`; `subdivCount` must match the draw count.
 * Scale `vertex` by radius and offset by object center for world position.
 */
export function icosphere(instanceIndex: number, vertexIndex: number, subdivCount: number) {
  'use gpu';
  const objectIndex = u32(instanceIndex / ICOSAHEDRON_FACE_COUNT);
  const faceIndex = u32(instanceIndex % ICOSAHEDRON_FACE_COUNT);
  const faceOffset = faceIndex * 3;
  const a = icosahedronFaceVertices.$[faceOffset] as v3f;
  const b = icosahedronFaceVertices.$[faceOffset + 1] as v3f;
  const c = icosahedronFaceVertices.$[faceOffset + 2] as v3f;
  return ProceduralSphereResult({
    instanceIndex: objectIndex,
    vertex: subdivSphericalTriangleSlot.$(a, b, c, vertexIndex, subdivCount),
  });
}
