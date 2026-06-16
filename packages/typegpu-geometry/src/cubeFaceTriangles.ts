import tgpu from 'typegpu';
import { arrayOf, type v3f, vec3f } from 'typegpu/data';

export const CUBE_FACE_COUNT = 6;
export const CUBE_FACE_TRIANGLE_COUNT = 12;

const cubeFaceQuads = [
  [vec3f(1, -1, -1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(1, 1, -1)],
  [vec3f(-1, -1, 1), vec3f(-1, -1, -1), vec3f(-1, 1, -1), vec3f(-1, 1, 1)],
  [vec3f(-1, 1, -1), vec3f(1, 1, -1), vec3f(1, 1, 1), vec3f(-1, 1, 1)],
  [vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, -1, -1), vec3f(-1, -1, -1)],
  [vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(-1, 1, 1)],
  [vec3f(1, -1, -1), vec3f(-1, -1, -1), vec3f(-1, 1, -1), vec3f(1, 1, -1)],
] as v3f[][];

const cubeFaceTrianglesData = cubeFaceQuads.flatMap(([bl, br, tr, tl]) => [bl, br, tr, bl, tr, tl]) as v3f[];

export const cubeFaceTriangles = tgpu.const(
  arrayOf(vec3f, cubeFaceTrianglesData.length),
  cubeFaceTrianglesData,
);
