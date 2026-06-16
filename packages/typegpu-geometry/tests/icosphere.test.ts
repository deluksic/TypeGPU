import { describe, expect, it } from 'vitest';
import {
  ICOSAHEDRON_FACE_COUNT,
  icosphereIndexCount,
  icosphereIndexCountPerFace,
  icosphereInstanceCount,
  icosphereWireframeIndexCount,
  icosphereWireframeIndexCountPerFace,
} from '../src/sphere/icosphere.ts';
import {
  subdivTriangleIndices,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../src/subdividedTriangle.ts';

const MAX_ICOSPHERE_SUBDIV = 10;

describe('icosphereIndexCount', () => {
  it('returns subdivisions² × 3 indices per face', () => {
    expect(icosphereIndexCountPerFace(4)).toBe(subdivTriangleIndexCount(4));
    expect(icosphereIndexCountPerFace(4)).toBe(48);
  });

  it('returns 20 × subdivisions² × 3 total indices', () => {
    expect(icosphereIndexCount(4)).toBe(960);
    expect(icosphereIndexCount(10)).toBe(6000);
  });
});

describe('icosphereWireframeIndexCount', () => {
  it('returns subdivisions² × 6 indices per face', () => {
    expect(icosphereWireframeIndexCountPerFace(4)).toBe(subdivTriangleWireframeIndexCount(4));
    expect(icosphereWireframeIndexCountPerFace(4)).toBe(96);
  });

  it('returns 20 × subdivisions² × 6 total indices', () => {
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
  it('uses subdivTriangleIndices as a per-face prefix at max subdiv', () => {
    const maxIndices = subdivTriangleIndices(MAX_ICOSPHERE_SUBDIV);

    for (let subdivisions = 1; subdivisions <= MAX_ICOSPHERE_SUBDIV; subdivisions++) {
      expect(maxIndices.slice(0, icosphereIndexCountPerFace(subdivisions))).toEqual(
        subdivTriangleIndices(subdivisions),
      );
    }
  });

  it('remaps draw instanceIndex to object index', () => {
    const objectCount = 300;
    const instanceCount = icosphereInstanceCount(objectCount);

    for (let drawInstanceIndex = 0; drawInstanceIndex < instanceCount; drawInstanceIndex += 137) {
      const objectIndex = Math.floor(drawInstanceIndex / ICOSAHEDRON_FACE_COUNT);
      expect(drawInstanceIndex % ICOSAHEDRON_FACE_COUNT).toBeLessThan(ICOSAHEDRON_FACE_COUNT);
      expect(objectIndex).toBeLessThan(objectCount);
    }
  });
});
