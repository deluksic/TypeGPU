import { f32, u32, vec3f, type v2f, type v3f } from 'typegpu/data';
import { sqrt } from 'typegpu/std';

/**
 * Index count for `segmentTriangleIndices(n)` (3 indices per tessellated triangle).
 * `n` is the number of equal segments along each triangle edge.
 */
export function segmentTriangleIndexCount(maxSegmentCount: number) {
  return maxSegmentCount * maxSegmentCount * 3;
}

/**
 * Index count for `segmentTriangleWireframeIndices(n)` (6 indices per tessellated triangle).
 */
export function segmentTriangleWireframeIndexCount(maxSegmentCount: number) {
  return maxSegmentCount * maxSegmentCount * 6;
}

/**
 * Triangle-list indices for a triangle lattice at `maxSegmentCount` segments per edge.
 *
 * The layout is shared by all procedural patch shapes in this package. Build once
 * at the maximum segment count, then draw a prefix per patch
 * (`segmentTriangleIndexCount(segmentCount)` indices) to control live tessellation.
 * Triangles beyond that prefix are left out of the draw, throwing away resolution
 * that isn't needed, so segment count can be changed per draw without uploading new indices.
 */
export function segmentTriangleIndices(maxSegmentCount: number) {
  const indices: number[] = [];
  for (let level = 0; level < maxSegmentCount; ++level) {
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

export function segmentTriangleWireframeIndices(maxSegmentCount: number) {
  const wireframeIndices: number[] = [];
  for (let level = 0; level < maxSegmentCount; ++level) {
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

export function segmentTriangleVertexCount(maxSegmentCount: number) {
  'use gpu';
  return u32((maxSegmentCount * (maxSegmentCount + 3)) / 2) + 1;
}

/** Barycentric weights (wA, wB, wC) for a vertex in the triangle segment lattice. */
export function triangleGridBarycentrics(vertexIndex: number, maxSegmentCount: number): v3f {
  'use gpu';
  const level = u32((sqrt(f32(vertexIndex << 3) + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) >> 1;
  const j = f32(vertexIndex - startIndex);
  const i = f32(level) - j;
  const invN = 1 / f32(maxSegmentCount);
  return vec3f(1 - (i + j) * invN, i * invN, j * invN);
}

export function segmentTriangle(
  A: v2f,
  B: v2f,
  C: v2f,
  vertexIndex: number,
  maxSegmentCount: number,
): v2f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, maxSegmentCount);
  return A * w.x + B * w.y + C * w.z;
}

export function segmentTriangle3(
  A: v3f,
  B: v3f,
  C: v3f,
  vertexIndex: number,
  maxSegmentCount: number,
): v3f {
  'use gpu';
  const w = triangleGridBarycentrics(vertexIndex, maxSegmentCount);
  return A * w.x + B * w.y + C * w.z;
}
