import { describe, expect, it } from 'vitest';
import {
  ROUNDED_BOX_PATCH_COUNT,
  roundedBoxIndexCount,
  roundedBoxIndexCountPerPatch,
  roundedBoxInstanceCount,
  roundedBoxWireframeIndexCount,
  roundedBoxWireframeIndexCountPerPatch,
} from '../src/box/roundedBox.ts';
import {
  subdivTriangleIndices,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../src/subdividedTriangle.ts';

const MAX_ROUNDED_BOX_SUBDIV = 10;

describe('roundedBoxIndexCount', () => {
  it('returns subdivisions² × 3 indices per patch', () => {
    expect(roundedBoxIndexCountPerPatch(4)).toBe(subdivTriangleIndexCount(4));
    expect(roundedBoxIndexCountPerPatch(4)).toBe(48);
  });

  it('returns 44 × subdivisions² × 3 total indices', () => {
    expect(roundedBoxIndexCount(4)).toBe(2112);
    expect(roundedBoxIndexCount(10)).toBe(13200);
  });
});

describe('roundedBoxWireframeIndexCount', () => {
  it('returns subdivisions² × 6 indices per patch', () => {
    expect(roundedBoxWireframeIndexCountPerPatch(4)).toBe(subdivTriangleWireframeIndexCount(4));
    expect(roundedBoxWireframeIndexCountPerPatch(4)).toBe(96);
  });

  it('returns 44 × subdivisions² × 6 total indices', () => {
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
  it('uses subdivTriangleIndices as a per-patch prefix at max subdiv', () => {
    const maxIndices = subdivTriangleIndices(MAX_ROUNDED_BOX_SUBDIV);

    for (let subdivisions = 1; subdivisions <= MAX_ROUNDED_BOX_SUBDIV; subdivisions++) {
      expect(maxIndices.slice(0, roundedBoxIndexCountPerPatch(subdivisions))).toEqual(
        subdivTriangleIndices(subdivisions),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = roundedBoxInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = Math.floor(drawInstanceIndex / ROUNDED_BOX_PATCH_COUNT);
      expect(drawInstanceIndex % ROUNDED_BOX_PATCH_COUNT).toBeLessThan(ROUNDED_BOX_PATCH_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
