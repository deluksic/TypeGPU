import { describe, expect, it } from 'vitest';
import {
  CUBE_FACE_TRIANGLE_COUNT,
  cubesphereIndexCount,
  cubesphereIndexCountPerFace,
  cubesphereInstanceCount,
  cubesphereWireframeIndexCount,
  cubesphereWireframeIndexCountPerFace,
} from '../src/sphere/cubesphere.ts';
import {
  subdivTriangleIndices,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../src/subdividedTriangle.ts';

const MAX_CUBESPHERE_SUBDIV = 10;

describe('cubesphereIndexCount', () => {
  it('returns subdivisions² × 3 indices per face triangle', () => {
    expect(cubesphereIndexCountPerFace(4)).toBe(subdivTriangleIndexCount(4));
    expect(cubesphereIndexCountPerFace(4)).toBe(48);
  });

  it('returns 12 × subdivisions² × 3 total indices', () => {
    expect(cubesphereIndexCount(4)).toBe(576);
    expect(cubesphereIndexCount(10)).toBe(3600);
  });
});

describe('cubesphereWireframeIndexCount', () => {
  it('returns subdivisions² × 6 indices per face triangle', () => {
    expect(cubesphereWireframeIndexCountPerFace(4)).toBe(subdivTriangleWireframeIndexCount(4));
    expect(cubesphereWireframeIndexCountPerFace(4)).toBe(96);
  });

  it('returns 12 × subdivisions² × 6 total indices', () => {
    expect(cubesphereWireframeIndexCount(4)).toBe(1152);
  });
});

describe('cubesphereInstanceCount', () => {
  it('returns objectCount × 12', () => {
    expect(cubesphereInstanceCount(300)).toBe(3600);
    expect(cubesphereInstanceCount(1)).toBe(CUBE_FACE_TRIANGLE_COUNT);
  });
});

describe('cubesphere instancing layout', () => {
  it('uses subdivTriangleIndices as a per-face prefix at max subdiv', () => {
    const maxIndices = subdivTriangleIndices(MAX_CUBESPHERE_SUBDIV);

    for (let subdivisions = 1; subdivisions <= MAX_CUBESPHERE_SUBDIV; subdivisions++) {
      expect(maxIndices.slice(0, cubesphereIndexCountPerFace(subdivisions))).toEqual(
        subdivTriangleIndices(subdivisions),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = cubesphereInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = Math.floor(drawInstanceIndex / CUBE_FACE_TRIANGLE_COUNT);
      expect(drawInstanceIndex % CUBE_FACE_TRIANGLE_COUNT).toBeLessThan(CUBE_FACE_TRIANGLE_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
