import tgpu from 'typegpu';
import { arrayOf, u32, type v3f, vec3f } from 'typegpu/data';
import { normalize } from 'typegpu/std';
import {
  subdivTriangle3,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../subdividedTriangle.ts';
import { ProceduralSphereResult } from './result.ts';
import { spherify } from './spherify.ts';

export const CUBE_FACE_TRIANGLE_COUNT = 12;

const cubeFaceQuads = [
  [vec3f(1, -1, -1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(1, 1, -1)],
  [vec3f(-1, -1, 1), vec3f(-1, -1, -1), vec3f(-1, 1, -1), vec3f(-1, 1, 1)],
  [vec3f(-1, 1, -1), vec3f(1, 1, -1), vec3f(1, 1, 1), vec3f(-1, 1, 1)],
  [vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, -1, -1), vec3f(-1, -1, -1)],
  [vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(-1, 1, 1)],
  [vec3f(1, -1, -1), vec3f(-1, -1, -1), vec3f(-1, 1, -1), vec3f(1, 1, -1)],
] as v3f[][];

const cubeFaceTrianglesData = cubeFaceQuads.flatMap(([bl, br, tr, tl]) => [bl, br, tr, bl, tr, tl]) as v3f[];

const cubeFaceTriangles = tgpu.const(
  arrayOf(vec3f, cubeFaceTrianglesData.length),
  cubeFaceTrianglesData,
);

export function cubesphereIndexCountPerFace(subdivisions: number) {
  return subdivTriangleIndexCount(subdivisions);
}

export function cubesphereIndexCount(subdivisions: number) {
  return CUBE_FACE_TRIANGLE_COUNT * cubesphereIndexCountPerFace(subdivisions);
}

export function cubesphereWireframeIndexCountPerFace(subdivisions: number) {
  return subdivTriangleWireframeIndexCount(subdivisions);
}

export function cubesphereWireframeIndexCount(subdivisions: number) {
  return CUBE_FACE_TRIANGLE_COUNT * cubesphereWireframeIndexCountPerFace(subdivisions);
}

export function cubesphereInstanceCount(objectCount: number) {
  return objectCount * CUBE_FACE_TRIANGLE_COUNT;
}

/**
 * Vertex shader helper for procedural cubespheres.
 *
 * Each cube face is two triangles (BL–BR–TR and BL–TR–TL). One instance per
 * face triangle (`cubesphereInstanceCount(n)`), shared `subdivTriangleIndices(maxSubdiv)`
 * index buffer, draw a prefix via `cubesphereIndexCountPerFace(subdivCount)`.
 * Vertices are linearly interpolated on the cube, then mapped with `spherify`.
 */
export function cubesphere(instanceIndex: number, vertexIndex: number, subdivCount: number) {
  'use gpu';
  const objectIndex = u32(instanceIndex / CUBE_FACE_TRIANGLE_COUNT);
  const faceTriangleIndex = u32(instanceIndex % CUBE_FACE_TRIANGLE_COUNT);
  const cornerOffset = faceTriangleIndex * 3;
  const a = cubeFaceTriangles.$[cornerOffset] as v3f;
  const b = cubeFaceTriangles.$[cornerOffset + 1] as v3f;
  const c = cubeFaceTriangles.$[cornerOffset + 2] as v3f;
  const onCube = subdivTriangle3(a, b, c, vertexIndex, subdivCount);
  return ProceduralSphereResult({
    instanceIndex: objectIndex,
    vertex: normalize(spherify(onCube)),
  });
}
