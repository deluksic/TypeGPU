import tgpu from 'typegpu';
import { arrayOf, f32, type v3f, vec3f } from 'typegpu/data';
import { select } from 'typegpu/std';

export const OCTAHEDRON_FACE_COUNT = 8;

export function octantSigns(octant: number): v3f {
  'use gpu';
  return vec3f(
    select(f32(1), f32(-1), (octant & 1) !== 0),
    select(f32(1), f32(-1), (octant & 2) !== 0),
    select(f32(1), f32(-1), (octant & 4) !== 0),
  );
}

export function edgeSign0(subIndex: number): number {
  'use gpu';
  return select(f32(1), f32(-1), (subIndex & 1) !== 0);
}

export function edgeSign1(subIndex: number): number {
  'use gpu';
  return select(f32(1), f32(-1), (subIndex & 2) !== 0);
}

const octahedronFaceVerticesData = Array.from({ length: OCTAHEDRON_FACE_COUNT }, (_, octant) => {
  const signs = octantSigns(octant);
  const a = vec3f(signs.x, 0, 0);
  const b = vec3f(0, signs.y, 0);
  const c = vec3f(0, 0, signs.z);
  if (signs.x * signs.y * signs.z > 0) {
    return [a, b, c];
  }
  return [a, c, b];
}).flat() as v3f[];

export const octahedronFaceVertices = tgpu.const(
  arrayOf(vec3f, octahedronFaceVerticesData.length),
  octahedronFaceVerticesData,
);
