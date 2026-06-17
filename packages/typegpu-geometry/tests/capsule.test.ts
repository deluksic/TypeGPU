import { describe, expect, it } from 'vitest';
import {
  CAPSULE_PATCH_COUNT,
  capsuleIndexCount,
  capsuleIndexCountPerPatch,
  capsuleInstanceCount,
  capsuleObjectIndex,
  capsulePatchIndex,
  capsuleWireframeIndexCount,
  capsuleWireframeIndexCountPerPatch,
} from '../src/capsule/capsule.ts';
import {
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndexCount,
} from '../src/segmentedTriangle.ts';

const MAX_CAPSULE_SEGMENT_COUNT = 10;

describe('capsuleIndexCount', () => {
  it('returns segmentCount² × 3 indices per patch', () => {
    expect(capsuleIndexCountPerPatch(4)).toBe(segmentTriangleIndexCount(4));
    expect(capsuleIndexCountPerPatch(4)).toBe(48);
  });

  it('returns 16 × segmentCount² × 3 total indices', () => {
    expect(capsuleIndexCount(4)).toBe(768);
    expect(capsuleIndexCount(10)).toBe(4800);
  });
});

describe('capsuleWireframeIndexCount', () => {
  it('returns segmentCount² × 6 indices per patch', () => {
    expect(capsuleWireframeIndexCountPerPatch(4)).toBe(segmentTriangleWireframeIndexCount(4));
    expect(capsuleWireframeIndexCountPerPatch(4)).toBe(96);
  });

  it('returns 16 × segmentCount² × 6 total indices', () => {
    expect(capsuleWireframeIndexCount(4)).toBe(1536);
  });
});

describe('capsuleInstanceCount', () => {
  it('returns objectCount × 16', () => {
    expect(capsuleInstanceCount(300)).toBe(4800);
    expect(capsuleInstanceCount(1)).toBe(CAPSULE_PATCH_COUNT);
  });
});

describe('capsule instancing layout', () => {
  it('uses segmentTriangleIndices as a per-patch prefix at max segment count', () => {
    const maxIndices = segmentTriangleIndices(MAX_CAPSULE_SEGMENT_COUNT);

    for (let segmentCount = 1; segmentCount <= MAX_CAPSULE_SEGMENT_COUNT; segmentCount++) {
      expect(maxIndices.slice(0, capsuleIndexCountPerPatch(segmentCount))).toEqual(
        segmentTriangleIndices(segmentCount),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = capsuleInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = capsuleObjectIndex(drawInstanceIndex);
      const patchIndex = capsulePatchIndex(drawInstanceIndex);
      expect(patchIndex).toBeLessThan(CAPSULE_PATCH_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
