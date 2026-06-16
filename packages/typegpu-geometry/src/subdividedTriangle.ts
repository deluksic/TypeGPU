import { f32, u32, type v2f } from 'typegpu/data';
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

export function subdivTriangle(
  A: v2f,
  B: v2f,
  C: v2f,
  vertexIndex: number,
  maxSubdivCount: number,
): v2f {
  'use gpu';
  const level = u32((sqrt(8 * f32(vertexIndex) + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) >> 1;
  const j = vertexIndex - startIndex;
  const i = level - j;
  const wB = f32(i) / f32(maxSubdivCount);
  const wC = f32(j) / f32(maxSubdivCount);
  const wA = f32(1) - wB - wC;
  return A * wA + B * wB + C * wC;
}
