import tgpu from 'typegpu';
import { u32, vec2f } from 'typegpu/data';
import type { v2f } from 'typegpu/data';
import { addMul, rot90ccw, rot90cw } from '../../utils.ts';
import { JoinPath, LineSegmentVertex } from '../types.ts';
import { add } from 'typegpu/std';

export const swallowtailCap = tgpu.fn([
  u32,
  JoinPath,
  LineSegmentVertex,
  vec2f,
  vec2f,
  vec2f,
  vec2f,
  vec2f,
], vec2f)(
  (
    vertexIndex,
    joinPath,
    V,
    vu,
    vd,
    right,
    dir,
    left,
  ) => {
    const dirRight = rot90cw(dir);
    const dirLeft = rot90ccw(dir);

    if (joinPath.depth >= 0) {
      const remove = [right, left];
      const dm = remove[joinPath.joinIndex & 0x1] as v2f;
      return addMul(V.position, dm, V.radius);
    }

    const v1 = addMul(V.position, add(dirRight, dir), V.radius);
    const v2 = addMul(V.position, vec2f(0, 0), 2 * V.radius);
    const v3 = addMul(V.position, add(dirLeft, dir), V.radius);
    const points = [vu, v1, v2, v3, vd];
    return points[vertexIndex % 5] as v2f;
  },
);
