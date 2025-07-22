import tgpu from 'typegpu';
import { JoinPath, LineSegmentVertex } from '../types.ts';
import { bool, u32, vec2f } from 'typegpu/data';
import type { v2f } from 'typegpu/data';
import { add, mul, normalize, select } from 'typegpu/std';
import { addMul, bisectCcw } from '../../utils.ts';
import { intersectLines, miterLimit, miterPoint } from '../utils.ts';

export const miterJoin = tgpu.fn([
  u32,
  u32,
  JoinPath,
  LineSegmentVertex,
  vec2f,
  vec2f,
  vec2f,
  vec2f,
  vec2f,
  vec2f,
  bool,
  bool,
], vec2f)(
  (
    situationIndex,
    vertexIndex,
    joinPath,
    V,
    vu,
    vd,
    ul,
    ur,
    dl,
    dr,
    joinU,
    joinD,
  ) => {
    const miterU = miterLimit(ur, ul);
    const miterD = miterLimit(dl, dr);
    const midR = bisectCcw(ur, dr);
    const midL = bisectCcw(dl, ul);

    const shouldCross = situationIndex === 1 || situationIndex === 4;
    const crossCenter = intersectLines(ul, dl, ur, dr).point;
    const averageCenter = mul(
      add(
        normalize(miterU.mid),
        normalize(miterD.mid),
      ),
      0.5,
    );

    let uR = ur;
    let u = miterU.mid;
    let c = select(averageCenter, crossCenter, shouldCross);
    let d = miterD.mid;
    let dR = dr;

    if (situationIndex === 2) {
      uR = ur;
      u = midR;
      c = midR;
      d = midR;
      dR = dr;
    }

    if (situationIndex === 3) {
      uR = ur;
      u = midL;
      c = midL;
      d = midL;
      dR = dr;
    }

    const joinIndex = joinPath.joinIndex;
    if (joinPath.depth >= 0) {
      const parents = [uR, u, d, dR];
      const d0 = parents[(joinIndex * 2) & 3] as v2f;
      const d1 = parents[(joinIndex * 2 + 1) & 3] as v2f;
      const dm = miterPoint(d0, d1);
      return addMul(V.position, dm, V.radius);
    }

    const v1 = select(vu, addMul(V.position, u, V.radius), joinU);
    const v2 = select(vu, addMul(V.position, c, V.radius), joinU || joinD);
    const v3 = select(vd, addMul(V.position, d, V.radius), joinD);
    const points = [vu, v1, v2, v3, vd];
    return points[vertexIndex % 5] as v2f;
  },
);
