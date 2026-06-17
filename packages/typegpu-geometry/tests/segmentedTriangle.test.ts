import { describe, expect, it } from 'vitest';
import {
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleVertexCount,
  segmentTriangleWireframeIndices,
  segmentTriangleWireframeIndexCount,
} from '../src/segmentedTriangle.ts';

describe('segmentTriangleIndices', () => {
  it('returns 012 for segment count 1', () => {
    expect(segmentTriangleIndices(1)).toEqual([0, 1, 2]);
  });

  it('returns 012 134 245 142 for segment count 2', () => {
    expect(segmentTriangleIndices(2)).toEqual([0, 1, 2, 1, 3, 4, 2, 4, 5, 1, 4, 2]);
  });

  it('produces maxSegmentCount^2 triangles', () => {
    for (const maxSegmentCount of [1, 2, 5, 10, 25]) {
      expect(segmentTriangleIndices(maxSegmentCount)).toHaveLength(
        segmentTriangleIndexCount(maxSegmentCount),
      );
    }
  });

  it('snapshot test for segment count 25', () => {
    expect(segmentTriangleIndices(25)).toMatchSnapshot();
  });
});

describe('segmentTriangleWireframeIndices', () => {
  it('returns line segments for segment count 1', () => {
    expect(segmentTriangleWireframeIndices(1)).toEqual([0, 1, 1, 2, 2, 0]);
  });

  it('returns line segments for segment count 2', () => {
    expect(segmentTriangleWireframeIndices(2)).toEqual([
      0, 1, 1, 2, 2, 0, 1, 3, 3, 4, 4, 1, 2, 4, 4, 5, 5, 2, 1, 4, 4, 2, 2, 1,
    ]);
  });

  it('produces maxSegmentCount^2 triangles worth of line segments', () => {
    for (const maxSegmentCount of [1, 2, 5, 10, 25]) {
      expect(segmentTriangleWireframeIndices(maxSegmentCount)).toHaveLength(
        segmentTriangleWireframeIndexCount(maxSegmentCount),
      );
    }
  });

  it('is a prefix of the level 10 wireframe indices', () => {
    const maxWireframeIndices = segmentTriangleWireframeIndices(10);
    for (let level = 1; level <= 10; level++) {
      expect(segmentTriangleWireframeIndices(level)).toEqual(
        maxWireframeIndices.slice(0, level * level * 6),
      );
    }
  });
});

function segmentTriangleCpu(
  A: [number, number],
  B: [number, number],
  C: [number, number],
  vertexIndex: number,
  maxSegmentCount: number,
): [number, number] {
  const level = Math.floor((Math.sqrt(8 * vertexIndex + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) / 2;
  const j = vertexIndex - startIndex;
  const i = level - j;
  const wB = i / maxSegmentCount;
  const wC = j / maxSegmentCount;
  const wA = 1 - wB - wC;
  return [A[0] * wA + B[0] * wB + C[0] * wC, A[1] * wA + B[1] * wB + C[1] * wC];
}

describe('segmentTriangle barycentric positions', () => {
  it('interpolates corners and interior points', () => {
    const A: [number, number] = [0.1, 0.2];
    const B: [number, number] = [0.9, 0.2];
    const C: [number, number] = [0.5, 0.9];

    expect(segmentTriangleCpu(A, B, C, 0, 1)).toEqual(A);
    expect(segmentTriangleCpu(A, B, C, 1, 1)).toEqual(B);
    expect(segmentTriangleCpu(A, B, C, 2, 1)[0]).toBeCloseTo(C[0]);
    expect(segmentTriangleCpu(A, B, C, 2, 1)[1]).toBeCloseTo(C[1]);
    expect(segmentTriangleCpu(A, B, C, 1, 4)[0]).toBeCloseTo(0.3);
    expect(segmentTriangleCpu(A, B, C, 1, 4)[1]).toBeCloseTo(0.2);
    expect(segmentTriangleCpu(A, B, C, 2, 4)[0]).toBeCloseTo(0.2);
    expect(segmentTriangleCpu(A, B, C, 2, 4)[1]).toBeCloseTo(0.375);
  });
});

describe('segmentTriangleVertexCount', () => {
  it('returns max index + 1 for segment count 1', () => {
    expect(segmentTriangleVertexCount(1)).toBe(3);
  });

  it('returns max index + 1 for segment count 2', () => {
    expect(segmentTriangleVertexCount(2)).toBe(6);
  });

  it('matches max(segmentTriangleIndices(n)) + 1', () => {
    for (const maxSegmentCount of [1, 2, 5, 10, 25]) {
      const indices = segmentTriangleIndices(maxSegmentCount);
      expect(segmentTriangleVertexCount(maxSegmentCount)).toBe(Math.max(...indices) + 1);
    }
  });
});
