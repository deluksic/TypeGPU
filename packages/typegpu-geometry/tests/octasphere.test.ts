import { describe, expect, it } from 'vitest';
import {
  OCTAHEDRON_FACE_COUNT,
  octasphereIndexCount,
  octasphereIndexCountPerFace,
  octasphereInstanceCount,
  octasphereObjectIndex,
  octaspherePatchIndex,
  octasphereWireframeIndexCount,
  octasphereWireframeIndexCountPerFace,
} from '../src/sphere/octasphere.ts';
import {
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndexCount,
} from '../src/segmentedTriangle.ts';

const MAX_OCTASPHERE_SEGMENT_COUNT = 10;

describe('octasphereIndexCount', () => {
  it('returns segmentCount² × 3 indices per face', () => {
    expect(octasphereIndexCountPerFace(4)).toBe(segmentTriangleIndexCount(4));
    expect(octasphereIndexCountPerFace(4)).toBe(48);
  });

  it('returns 8 × segmentCount² × 3 total indices', () => {
    expect(octasphereIndexCount(4)).toBe(384);
    expect(octasphereIndexCount(10)).toBe(2400);
  });
});

describe('octasphereWireframeIndexCount', () => {
  it('returns segmentCount² × 6 indices per face', () => {
    expect(octasphereWireframeIndexCountPerFace(4)).toBe(segmentTriangleWireframeIndexCount(4));
    expect(octasphereWireframeIndexCountPerFace(4)).toBe(96);
  });

  it('returns 8 × segmentCount² × 6 total indices', () => {
    expect(octasphereWireframeIndexCount(4)).toBe(768);
  });
});

describe('octasphereInstanceCount', () => {
  it('returns objectCount × 8', () => {
    expect(octasphereInstanceCount(300)).toBe(2400);
    expect(octasphereInstanceCount(1)).toBe(OCTAHEDRON_FACE_COUNT);
  });
});

describe('octasphere instancing layout', () => {
  it('uses segmentTriangleIndices as a per-face prefix at max segment count', () => {
    const maxIndices = segmentTriangleIndices(MAX_OCTASPHERE_SEGMENT_COUNT);

    for (let segmentCount = 1; segmentCount <= MAX_OCTASPHERE_SEGMENT_COUNT; segmentCount++) {
      expect(maxIndices.slice(0, octasphereIndexCountPerFace(segmentCount))).toEqual(
        segmentTriangleIndices(segmentCount),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = octasphereInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = octasphereObjectIndex(drawInstanceIndex);
      const patchIndex = octaspherePatchIndex(drawInstanceIndex);
      expect(patchIndex).toBeLessThan(OCTAHEDRON_FACE_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
