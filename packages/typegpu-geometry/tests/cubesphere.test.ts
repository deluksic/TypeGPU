import { describe, expect, it } from 'vitest';
import {
  CUBE_FACE_TRIANGLE_COUNT,
  cubesphereIndexCount,
  cubesphereIndexCountPerFace,
  cubesphereInstanceCount,
  cubesphereObjectIndex,
  cubespherePatchIndex,
  cubesphereWireframeIndexCount,
  cubesphereWireframeIndexCountPerFace,
} from '../src/sphere/cubesphere.ts';
import {
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndexCount,
} from '../src/segmentedTriangle.ts';

const MAX_CUBESPHERE_SEGMENT_COUNT = 10;

describe('cubesphereIndexCount', () => {
  it('returns segmentCount² × 3 indices per face triangle', () => {
    expect(cubesphereIndexCountPerFace(4)).toBe(segmentTriangleIndexCount(4));
    expect(cubesphereIndexCountPerFace(4)).toBe(48);
  });

  it('returns 12 × segmentCount² × 3 total indices', () => {
    expect(cubesphereIndexCount(4)).toBe(576);
    expect(cubesphereIndexCount(10)).toBe(3600);
  });
});

describe('cubesphereWireframeIndexCount', () => {
  it('returns segmentCount² × 6 indices per face triangle', () => {
    expect(cubesphereWireframeIndexCountPerFace(4)).toBe(segmentTriangleWireframeIndexCount(4));
    expect(cubesphereWireframeIndexCountPerFace(4)).toBe(96);
  });

  it('returns 12 × segmentCount² × 6 total indices', () => {
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
  it('uses segmentTriangleIndices as a per-face prefix at max segment count', () => {
    const maxIndices = segmentTriangleIndices(MAX_CUBESPHERE_SEGMENT_COUNT);

    for (let segmentCount = 1; segmentCount <= MAX_CUBESPHERE_SEGMENT_COUNT; segmentCount++) {
      expect(maxIndices.slice(0, cubesphereIndexCountPerFace(segmentCount))).toEqual(
        segmentTriangleIndices(segmentCount),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = cubesphereInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = cubesphereObjectIndex(drawInstanceIndex);
      const patchIndex = cubespherePatchIndex(drawInstanceIndex);
      expect(patchIndex).toBeLessThan(CUBE_FACE_TRIANGLE_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
