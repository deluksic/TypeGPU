import tgpu from 'typegpu';
import { arrayOf, u32, type v3f, vec3f } from 'typegpu/data';
import { subdivSphericalTriangleSlot } from './subdivSphericalTriangle/slots.ts';
import {
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../subdividedTriangle.ts';
import { ProceduralSphereResult } from './result.ts';

export const OCTAHEDRON_FACE_COUNT = 8;

function octantSigns(octant: number): [number, number, number] {
  return [
    (octant & 1) !== 0 ? -1 : 1,
    (octant & 2) !== 0 ? -1 : 1,
    (octant & 4) !== 0 ? -1 : 1,
  ];
}

const octahedronFaceVerticesData = Array.from({ length: OCTAHEDRON_FACE_COUNT }, (_, octant) => {
  const [sx, sy, sz] = octantSigns(octant);
  const a = vec3f(sx, 0, 0);
  const b = vec3f(0, sy, 0);
  const c = vec3f(0, 0, sz);
  if (sx * sy * sz > 0) {
    return [a, b, c];
  }
  return [a, c, b];
}).flat() as v3f[];

const octahedronFaceVertices = tgpu.const(
  arrayOf(vec3f, octahedronFaceVerticesData.length),
  octahedronFaceVerticesData,
);

export function octasphereIndexCountPerFace(subdivisions: number) {
  return subdivTriangleIndexCount(subdivisions);
}

export function octasphereIndexCount(subdivisions: number) {
  return OCTAHEDRON_FACE_COUNT * octasphereIndexCountPerFace(subdivisions);
}

export function octasphereWireframeIndexCountPerFace(subdivisions: number) {
  return subdivTriangleWireframeIndexCount(subdivisions);
}

export function octasphereWireframeIndexCount(subdivisions: number) {
  return OCTAHEDRON_FACE_COUNT * octasphereWireframeIndexCountPerFace(subdivisions);
}

export function octasphereInstanceCount(objectCount: number) {
  return objectCount * OCTAHEDRON_FACE_COUNT;
}

/**
 * Vertex shader helper for procedural octaspheres.
 *
 * Setup: one instance per octahedron face (`octasphereInstanceCount(n)`), shared
 * `subdivTriangleIndices(maxSubdiv)` index buffer, draw a prefix via `octasphereIndexCountPerFace(subdivCount)`.
 * Pass `@builtin(instance_index)` / `@builtin(vertex_index)`; `subdivCount` must match the draw count.
 * Scale `vertex` by radius and offset by object center for world position.
 */
export function octasphere(instanceIndex: number, vertexIndex: number, subdivCount: number) {
  'use gpu';
  const objectIndex = u32(instanceIndex / OCTAHEDRON_FACE_COUNT);
  const faceIndex = u32(instanceIndex % OCTAHEDRON_FACE_COUNT);
  const faceOffset = faceIndex * 3;
  const a = octahedronFaceVertices.$[faceOffset] as v3f;
  const b = octahedronFaceVertices.$[faceOffset + 1] as v3f;
  const c = octahedronFaceVertices.$[faceOffset + 2] as v3f;
  const vertex = subdivSphericalTriangleSlot.$(a, b, c, vertexIndex, subdivCount);
  return ProceduralSphereResult({
    instanceIndex: objectIndex,
    vertex,
    normal: vertex,
  });
}
