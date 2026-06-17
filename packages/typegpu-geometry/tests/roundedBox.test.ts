import { describe, expect, it } from 'vitest';
import {
  ROUNDED_BOX_PATCH_COUNT,
  roundedBoxIndexCount,
  roundedBoxIndexCountPerPatch,
  roundedBoxInstanceCount,
  roundedBoxObjectIndex,
  roundedBoxPatchIndex,
  roundedBoxWireframeIndexCount,
  roundedBoxWireframeIndexCountPerPatch,
} from '../src/box/roundedBox.ts';
import {
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndexCount,
} from '../src/segmentedTriangle.ts';

const MAX_ROUNDED_BOX_SEGMENT_COUNT = 10;

describe('roundedBoxIndexCount', () => {
  it('returns segmentCount² × 3 indices per patch', () => {
    expect(roundedBoxIndexCountPerPatch(4)).toBe(segmentTriangleIndexCount(4));
    expect(roundedBoxIndexCountPerPatch(4)).toBe(48);
  });

  it('returns 44 × segmentCount² × 3 total indices', () => {
    expect(roundedBoxIndexCount(4)).toBe(2112);
    expect(roundedBoxIndexCount(10)).toBe(13200);
  });
});

describe('roundedBoxWireframeIndexCount', () => {
  it('returns segmentCount² × 6 indices per patch', () => {
    expect(roundedBoxWireframeIndexCountPerPatch(4)).toBe(segmentTriangleWireframeIndexCount(4));
    expect(roundedBoxWireframeIndexCountPerPatch(4)).toBe(96);
  });

  it('returns 44 × segmentCount² × 6 total indices', () => {
    expect(roundedBoxWireframeIndexCount(4)).toBe(4224);
  });
});

describe('roundedBoxInstanceCount', () => {
  it('returns objectCount × 44', () => {
    expect(roundedBoxInstanceCount(300)).toBe(13200);
    expect(roundedBoxInstanceCount(1)).toBe(ROUNDED_BOX_PATCH_COUNT);
  });
});

describe('roundedBox instancing layout', () => {
  it('uses segmentTriangleIndices as a per-patch prefix at max segment count', () => {
    const maxIndices = segmentTriangleIndices(MAX_ROUNDED_BOX_SEGMENT_COUNT);

    for (let segmentCount = 1; segmentCount <= MAX_ROUNDED_BOX_SEGMENT_COUNT; segmentCount++) {
      expect(maxIndices.slice(0, roundedBoxIndexCountPerPatch(segmentCount))).toEqual(
        segmentTriangleIndices(segmentCount),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = roundedBoxInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = roundedBoxObjectIndex(drawInstanceIndex);
      const patchIndex = roundedBoxPatchIndex(drawInstanceIndex);
      expect(patchIndex).toBeLessThan(ROUNDED_BOX_PATCH_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
