import { describe, expect, it } from 'vitest';
import {
  ICOSAHEDRON_FACE_COUNT,
  icosphereIndexCount,
  icosphereIndexCountPerFace,
  icosphereInstanceCount,
  icosphereObjectIndex,
  icospherePatchIndex,
  icosphereWireframeIndexCount,
  icosphereWireframeIndexCountPerFace,
} from '../src/sphere/icosphere.ts';
import {
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndexCount,
} from '../src/segmentedTriangle.ts';

const MAX_ICOSPHERE_SEGMENT_COUNT = 10;

describe('icosphereIndexCount', () => {
  it('returns segmentCount² × 3 indices per face', () => {
    expect(icosphereIndexCountPerFace(4)).toBe(segmentTriangleIndexCount(4));
    expect(icosphereIndexCountPerFace(4)).toBe(48);
  });

  it('returns 20 × segmentCount² × 3 total indices', () => {
    expect(icosphereIndexCount(4)).toBe(960);
    expect(icosphereIndexCount(10)).toBe(6000);
  });
});

describe('icosphereWireframeIndexCount', () => {
  it('returns segmentCount² × 6 indices per face', () => {
    expect(icosphereWireframeIndexCountPerFace(4)).toBe(segmentTriangleWireframeIndexCount(4));
    expect(icosphereWireframeIndexCountPerFace(4)).toBe(96);
  });

  it('returns 20 × segmentCount² × 6 total indices', () => {
    expect(icosphereWireframeIndexCount(4)).toBe(1920);
  });
});

describe('icosphereInstanceCount', () => {
  it('returns objectCount × 20', () => {
    expect(icosphereInstanceCount(300)).toBe(6000);
    expect(icosphereInstanceCount(1)).toBe(ICOSAHEDRON_FACE_COUNT);
  });
});

describe('icosphere instancing layout', () => {
  it('uses segmentTriangleIndices as a per-face prefix at max segment count', () => {
    const maxIndices = segmentTriangleIndices(MAX_ICOSPHERE_SEGMENT_COUNT);

    for (let segmentCount = 1; segmentCount <= MAX_ICOSPHERE_SEGMENT_COUNT; segmentCount++) {
      expect(maxIndices.slice(0, icosphereIndexCountPerFace(segmentCount))).toEqual(
        segmentTriangleIndices(segmentCount),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = icosphereInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = icosphereObjectIndex(drawInstanceIndex);
      const patchIndex = icospherePatchIndex(drawInstanceIndex);
      expect(patchIndex).toBeLessThan(ICOSAHEDRON_FACE_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
