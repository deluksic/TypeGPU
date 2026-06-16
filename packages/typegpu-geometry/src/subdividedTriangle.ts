import { f32, u32, vec3f, type v2f, type v3f } from 'typegpu/data';
import { sqrt } from 'typegpu/std';

/**
 * Index count for `subdivTriangleIndices(n)` (3 indices per subdivided triangle).
 */
export function subdivTriangleIndexCount(maxSubdivCount: number) {
  return maxSubdivCount * maxSubdivCount * 3;
}

/**
 * Index count for `subdivTriangleWireframeIndices(n)` (6 indices per subdivided triangle).
 */
export function subdivTriangleWireframeIndexCount(maxSubdivCount: number) {
  return maxSubdivCount * maxSubdivCount * 6;
}

export function subdivTriangleIndices(maxSubdivCount: number) {
  const indices: number[] = [];
  for (let level = 0; level < maxSubdivCount; ++level) {
    const startIndex = (level * (level + 1)) / 2;
    for (let i = 0; i < level + 1; ++i) {
      const index = startIndex + i;
      indices.push(index, index + level + 1, index + level + 2);
    }
    for (let i = 0; i < level; ++i) {
      const index = startIndex + i;
      indices.push(index, index + level + 2, index + 1);
    }
  }
  return indices;
}

export function subdivTriangleWireframeIndices(maxSubdivCount: number) {
  const wireframeIndices: number[] = [];
  for (let level = 0; level < maxSubdivCount; ++level) {
    const startIndex = (level * (level + 1)) / 2;
    for (let i = 0; i < level + 1; ++i) {
      const index = startIndex + i;
      const a = index;
      const b = index + level + 1;
      const c = index + level + 2;
      wireframeIndices.push(a, b, b, c, c, a);
    }
    for (let i = 0; i < level; ++i) {
      const index = startIndex + i;
      const a = index;
      const b = index + level + 2;
      const c = index + 1;
      wireframeIndices.push(a, b, b, c, c, a);
    }
  }
  return wireframeIndices;
}

export function subdivTriangleVertexCount(maxSubdivCount: number) {
  'use gpu';
  return u32((maxSubdivCount * (maxSubdivCount + 3)) / 2) + 1;
}

/** Barycentric weights (wA, wB, wC) for a vertex in the subdivided triangle lattice. */
export function triangleGridBarycentrics(vertexIndex: number, maxSubdivCount: number): v3f {
  'use gpu';
  const level = u32((sqrt(8 * f32(vertexIndex) + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) >> 1;
  const j = f32(vertexIndex - startIndex);
  const i = f32(level) - j;
  const invN = f32(1) / f32(maxSubdivCount);
  return vec3f(f32(1) - (i + j) * invN, i * invN, j * invN);
}

export function subdivTriangle(
  A: v2f,
  B: v2f,
  C: v2f,
  vertexIndex: number,
  maxSubdivCount: number,
): v2f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, maxSubdivCount);
  return A * w.x + B * w.y + C * w.z;
}

export function subdivTriangle3(
  A: v3f,
  B: v3f,
  C: v3f,
  vertexIndex: number,
  maxSubdivCount: number,
): v3f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, maxSubdivCount);
  return A * w.x + B * w.y + C * w.z;
}
