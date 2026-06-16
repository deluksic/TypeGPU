import { describe, expect, it } from 'vitest';
import {
  CAPSULE_PATCH_COUNT,
  capsuleIndexCount,
  capsuleIndexCountPerPatch,
  capsuleInstanceCount,
  capsuleWireframeIndexCount,
  capsuleWireframeIndexCountPerPatch,
} from '../src/capsule/capsule.ts';
import {
  subdivTriangleIndices,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../src/subdividedTriangle.ts';

const MAX_CAPSULE_SUBDIV = 10;

describe('capsuleIndexCount', () => {
  it('returns subdivisions² × 3 indices per patch', () => {
    expect(capsuleIndexCountPerPatch(4)).toBe(subdivTriangleIndexCount(4));
    expect(capsuleIndexCountPerPatch(4)).toBe(48);
  });

  it('returns 16 × subdivisions² × 3 total indices', () => {
    expect(capsuleIndexCount(4)).toBe(768);
    expect(capsuleIndexCount(10)).toBe(4800);
  });
});

describe('capsuleWireframeIndexCount', () => {
  it('returns subdivisions² × 6 indices per patch', () => {
    expect(capsuleWireframeIndexCountPerPatch(4)).toBe(subdivTriangleWireframeIndexCount(4));
    expect(capsuleWireframeIndexCountPerPatch(4)).toBe(96);
  });

  it('returns 16 × subdivisions² × 6 total indices', () => {
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
  it('uses subdivTriangleIndices as a per-patch prefix at max subdiv', () => {
    const maxIndices = subdivTriangleIndices(MAX_CAPSULE_SUBDIV);

    for (let subdivisions = 1; subdivisions <= MAX_CAPSULE_SUBDIV; subdivisions++) {
      expect(maxIndices.slice(0, capsuleIndexCountPerPatch(subdivisions))).toEqual(
        subdivTriangleIndices(subdivisions),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = capsuleInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = Math.floor(drawInstanceIndex / CAPSULE_PATCH_COUNT);
      expect(drawInstanceIndex % CAPSULE_PATCH_COUNT).toBeLessThan(CAPSULE_PATCH_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
