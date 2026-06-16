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

function slerpApproxWarpCpu(d: number, t: number) {
  const cosHalfAngle = Math.sqrt(0.5 * (1 + d));
  const B = (2 - 2 * cosHalfAngle) / (2 + cosHalfAngle);
  const C1 = (1 - B) * 0.5;
  const u = t + t - 1;
  const u2 = u * u;
  return (u * C1) / (1 - B * u2) + 0.5;
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
}

function dot3(a: [number, number, number], b: [number, number, number]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function warpAlongABCpu(w: [number, number, number], d: number): [number, number, number] {
  const s = w[0] + w[1];
  const u = slerpApproxWarpCpu(d, w[1] / s);
  return [(1 - u) * s, u * s, w[2]];
}

function warpAlongACCpu(w: [number, number, number], d: number): [number, number, number] {
  const s = w[0] + w[2];
  const u = slerpApproxWarpCpu(d, w[2] / s);
  return [(1 - u) * s, w[1], u * s];
}

function warpAlongBCCpu(w: [number, number, number], d: number): [number, number, number] {
  const s = w[1] + w[2];
  const u = slerpApproxWarpCpu(d, w[2] / s);
  return [w[0], (1 - u) * s, u * s];
}

function barycentricOnSphereCpu(
  w: [number, number, number],
  A: [number, number, number],
  B: [number, number, number],
  C: [number, number, number],
): [number, number, number] {
  return normalize3([
    A[0] * w[0] + B[0] * w[1] + C[0] * w[2],
    A[1] * w[0] + B[1] * w[1] + C[1] * w[2],
    A[2] * w[0] + B[2] * w[1] + C[2] * w[2],
  ]);
}

function subdivSphericalTriangleCpu(
  A: [number, number, number],
  B: [number, number, number],
  C: [number, number, number],
  vertexIndex: number,
  maxSubdivCount: number,
): [number, number, number] {
  const level = Math.floor((Math.sqrt(8 * vertexIndex + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) / 2;
  const j = vertexIndex - startIndex;
  const i = level - j;
  const invN = 1 / maxSubdivCount;
  const w: [number, number, number] = [1 - (i + j) * invN, i * invN, j * invN];
  const edgeDots: [number, number, number] = [dot3(A, B), dot3(A, C), dot3(B, C)];
  const w1 = warpAlongABCpu(w, edgeDots[0]);
  const w2 = warpAlongACCpu(w, edgeDots[1]);
  const w3 = warpAlongBCCpu(w, edgeDots[2]);
  const blend: [number, number, number] = [w[0] * w[1], w[0] * w[2], w[1] * w[2]];
  const wSum = blend[0] + blend[1] + blend[2];
  const linear = barycentricOnSphereCpu(w, A, B, C);
  let warped = linear;
  if (wSum > 1e-6) {
    const wFinal: [number, number, number] = [
      (w1[0] * blend[0] + w2[0] * blend[1] + w3[0] * blend[2]) / wSum,
      (w1[1] * blend[0] + w2[1] * blend[1] + w3[1] * blend[2]) / wSum,
      (w1[2] * blend[0] + w2[2] * blend[1] + w3[2] * blend[2]) / wSum,
    ];
    warped = barycentricOnSphereCpu(wFinal, A, B, C);
  }
  const center = normalize3([A[0] + B[0] + C[0], A[1] + B[1] + C[1], A[2] + B[2] + C[2]]);
  const t = Math.min(w[0], w[1], w[2]) * 0.25;
  const tw = slerpApproxWarpCpu(dot3(warped, center), t);
  return normalize3([
    warped[0] * (1 - tw) + center[0] * tw,
    warped[1] * (1 - tw) + center[1] * tw,
    warped[2] * (1 - tw) + center[2] * tw,
  ]);
}

function barycentricWeights(vertexIndex: number, maxSubdivCount: number) {
  const level = Math.floor((Math.sqrt(8 * vertexIndex + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) / 2;
  const j = vertexIndex - startIndex;
  const i = level - j;
  const wB = i / maxSubdivCount;
  const wC = j / maxSubdivCount;
  const wA = 1 - wB - wC;
  return { wA, wB, wC, level, i, j };
}

describe('subdivSphericalTriangle seam safety', () => {
  const goldenRatio = (1 + Math.sqrt(5)) / 2;
  const icosahedronVertices = (
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
  ).map(([x, y, z]) => normalize3([x, y, z]));

  const face0 = [0, 11, 5].map((i) => icosahedronVertices[i]) as [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  const face1 = [0, 5, 1].map((i) => icosahedronVertices[i]) as [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];

  it('matches on shared edge between adjacent icosahedron faces', () => {
    const n = 8;
    for (let vertexIndex0 = 0; vertexIndex0 < subdivTriangleVertexCount(n); vertexIndex0++) {
      const w0 = barycentricWeights(vertexIndex0, n);
      if (w0.wB !== 0) continue;
      const p0 = subdivSphericalTriangleCpu(face0[0], face0[1], face0[2], vertexIndex0, n);
      for (let vertexIndex1 = 0; vertexIndex1 < subdivTriangleVertexCount(n); vertexIndex1++) {
        const w1 = barycentricWeights(vertexIndex1, n);
        if (w1.wC !== 0) continue;
        if (w1.wA !== w0.wA || w1.wB !== w0.wC) continue;
        const p1 = subdivSphericalTriangleCpu(face1[0], face1[1], face1[2], vertexIndex1, n);
        expect(p1[0]).toBeCloseTo(p0[0], 5);
        expect(p1[1]).toBeCloseTo(p0[1], 5);
        expect(p1[2]).toBeCloseTo(p0[2], 5);
      }
    }
  });
});
