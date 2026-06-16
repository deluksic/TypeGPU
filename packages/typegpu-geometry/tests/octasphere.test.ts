import { describe, expect, it } from 'vitest';
import {
  OCTAHEDRON_FACE_COUNT,
  octasphereIndexCount,
  octasphereIndexCountPerFace,
  octasphereInstanceCount,
  octasphereWireframeIndexCount,
  octasphereWireframeIndexCountPerFace,
} from '../src/sphere/octasphere.ts';
import {
  subdivTriangleIndices,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../src/subdividedTriangle.ts';

const MAX_OCTASPHERE_SUBDIV = 10;

describe('octasphereIndexCount', () => {
  it('returns subdivisions² × 3 indices per face', () => {
    expect(octasphereIndexCountPerFace(4)).toBe(subdivTriangleIndexCount(4));
    expect(octasphereIndexCountPerFace(4)).toBe(48);
  });

  it('returns 8 × subdivisions² × 3 total indices', () => {
    expect(octasphereIndexCount(4)).toBe(384);
    expect(octasphereIndexCount(10)).toBe(2400);
  });
});

describe('octasphereWireframeIndexCount', () => {
  it('returns subdivisions² × 6 indices per face', () => {
    expect(octasphereWireframeIndexCountPerFace(4)).toBe(subdivTriangleWireframeIndexCount(4));
    expect(octasphereWireframeIndexCountPerFace(4)).toBe(96);
  });

  it('returns 8 × subdivisions² × 6 total indices', () => {
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
  it('uses subdivTriangleIndices as a per-face prefix at max subdiv', () => {
    const maxIndices = subdivTriangleIndices(MAX_OCTASPHERE_SUBDIV);

    for (let subdivisions = 1; subdivisions <= MAX_OCTASPHERE_SUBDIV; subdivisions++) {
      expect(maxIndices.slice(0, octasphereIndexCountPerFace(subdivisions))).toEqual(
        subdivTriangleIndices(subdivisions),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = octasphereInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = Math.floor(drawInstanceIndex / OCTAHEDRON_FACE_COUNT);
      expect(drawInstanceIndex % OCTAHEDRON_FACE_COUNT).toBeLessThan(OCTAHEDRON_FACE_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
