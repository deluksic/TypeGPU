import { describe, expect, it } from 'vitest';
import {
  subdivTriangleIndices,
  subdivTriangleIndexCount,
  subdivTriangleVertexCount,
  subdivTriangleWireframeIndices,
  subdivTriangleWireframeIndexCount,
} from '../src/subdividedTriangle.ts';

describe('subdivTriangleIndices', () => {
  it('returns 012 for subdiv level 1', () => {
    expect(subdivTriangleIndices(1)).toEqual([0, 1, 2]);
  });

  it('returns 012 134 245 142 for subdiv level 2', () => {
    expect(subdivTriangleIndices(2)).toEqual([0, 1, 2, 1, 3, 4, 2, 4, 5, 1, 4, 2]);
  });

  it('produces maxSubdivCount^2 triangles', () => {
    for (const maxSubdivCount of [1, 2, 5, 10, 25]) {
      expect(subdivTriangleIndices(maxSubdivCount)).toHaveLength(
        subdivTriangleIndexCount(maxSubdivCount),
      );
    }
  });

  it('snapshot test for subdiv level 25', () => {
    expect(subdivTriangleIndices(25)).toMatchSnapshot();
  });
});

describe('subdivTriangleWireframeIndices', () => {
  it('returns line segments for subdiv level 1', () => {
    expect(subdivTriangleWireframeIndices(1)).toEqual([0, 1, 1, 2, 2, 0]);
  });

  it('returns line segments for subdiv level 2', () => {
    expect(subdivTriangleWireframeIndices(2)).toEqual([
      0, 1, 1, 2, 2, 0, 1, 3, 3, 4, 4, 1, 2, 4, 4, 5, 5, 2, 1, 4, 4, 2, 2, 1,
    ]);
  });

  it('produces maxSubdivCount^2 triangles worth of line segments', () => {
    for (const maxSubdivCount of [1, 2, 5, 10, 25]) {
      expect(subdivTriangleWireframeIndices(maxSubdivCount)).toHaveLength(
        subdivTriangleWireframeIndexCount(maxSubdivCount),
      );
    }
  });

  it('is a prefix of the level 10 wireframe indices', () => {
    const maxWireframeIndices = subdivTriangleWireframeIndices(10);
    for (let level = 1; level <= 10; level++) {
      expect(subdivTriangleWireframeIndices(level)).toEqual(
        maxWireframeIndices.slice(0, level * level * 6),
      );
    }
  });
});

function subdivTriangleCpu(
  A: [number, number],
  B: [number, number],
  C: [number, number],
  vertexIndex: number,
  maxSubdivCount: number,
): [number, number] {
  const level = Math.floor((Math.sqrt(8 * vertexIndex + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) / 2;
  const j = vertexIndex - startIndex;
  const i = level - j;
  const wB = i / maxSubdivCount;
  const wC = j / maxSubdivCount;
  const wA = 1 - wB - wC;
  return [A[0] * wA + B[0] * wB + C[0] * wC, A[1] * wA + B[1] * wB + C[1] * wC];
}

describe('subdivTriangle barycentric positions', () => {
  it('interpolates corners and interior points', () => {
    const A: [number, number] = [0.1, 0.2];
    const B: [number, number] = [0.9, 0.2];
    const C: [number, number] = [0.5, 0.9];

    expect(subdivTriangleCpu(A, B, C, 0, 1)).toEqual(A);
    expect(subdivTriangleCpu(A, B, C, 1, 1)).toEqual(B);
    expect(subdivTriangleCpu(A, B, C, 2, 1)[0]).toBeCloseTo(C[0]);
    expect(subdivTriangleCpu(A, B, C, 2, 1)[1]).toBeCloseTo(C[1]);
    expect(subdivTriangleCpu(A, B, C, 1, 4)[0]).toBeCloseTo(0.3);
    expect(subdivTriangleCpu(A, B, C, 1, 4)[1]).toBeCloseTo(0.2);
    expect(subdivTriangleCpu(A, B, C, 2, 4)[0]).toBeCloseTo(0.2);
    expect(subdivTriangleCpu(A, B, C, 2, 4)[1]).toBeCloseTo(0.375);
  });
});

describe('subdivTriangleVertexCount', () => {
  it('returns max index + 1 for subdiv level 1', () => {
    expect(subdivTriangleVertexCount(1)).toBe(3);
  });

  it('returns max index + 1 for subdiv level 2', () => {
    expect(subdivTriangleVertexCount(2)).toBe(6);
  });

  it('matches max(subdivTriangleIndices(n)) + 1', () => {
    for (const maxSubdivCount of [1, 2, 5, 10, 25]) {
      const indices = subdivTriangleIndices(maxSubdivCount);
      expect(subdivTriangleVertexCount(maxSubdivCount)).toBe(Math.max(...indices) + 1);
    }
  });
});
