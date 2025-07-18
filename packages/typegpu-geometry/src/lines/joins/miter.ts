import tgpu from 'typegpu';
import { f32, vec2f } from 'typegpu/data';
import { add, dot, normalize, select, sub } from 'typegpu/std';
import { bisectCcw, cross2d, midPoint } from '../../utils.ts';
import { JoinResult } from '../types.ts';
import { miterLimit, miterPoint } from '../utils.ts';

export const miterJoin = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f, vec2f],
  JoinResult,
)(
  (nL, nUL, nUR, nDL, nDR) => {
    const xL = dot(nL, nUL); // == dot(nL, nDL)
    const xUR = dot(nL, nUR);
    const xDR = dot(nL, nDR);
    const yUR = cross2d(nL, nUR);
    const yDR = cross2d(nL, nDR);
    const sideUR = select(f32(-1), f32(1), xUR - xL >= 0);
    const sideDR = select(f32(-1), f32(1), xDR - xL >= 0);

    const midR = bisectCcw(nUR, nDR);
    const midL = bisectCcw(nDL, nUL);
    const reverseMiterU = miterPoint(nUL, nUR);
    const reverseMiterD = miterPoint(nDR, nDL);

    const miterU = miterLimit(nUR, nUL);
    const miterD = miterLimit(nDL, nDR);

    if (sideUR === sideDR) {
      const side = sideUR;
      const XUR = select(xUR, side * 2 - xUR, yUR < 0);
      const XDR = select(xDR, side * 2 - xDR, yDR < 0);
      const clockWise = (side * (XUR - XDR)) <= 0;
      if (side === 1) {
        if (clockWise) {
          return JoinResult({
            uL: miterU.left,
            u: miterU.mid,
            uR: miterU.right,
            c: midPoint(miterU.mid, miterD.mid),
            dL: miterD.left,
            d: miterD.mid,
            dR: miterD.right,
            joinUL: false,
            joinUR: false,
            joinDL: false,
            joinDR: false,
            situationIndex: 0,
          });
        }
        return JoinResult({
          uL: nUL,
          u: midR,
          uR: nUR,
          c: midR,
          dL: nDL,
          d: midR,
          dR: nDR,
          joinUL: false,
          joinUR: false,
          joinDL: false,
          joinDR: false,
          situationIndex: 1,
        });
      }
      // side == -1
      if (clockWise) {
        return JoinResult({
          uL: reverseMiterU,
          u: reverseMiterU,
          uR: reverseMiterU,
          c: midPoint(reverseMiterU, reverseMiterD),
          dL: reverseMiterD,
          d: reverseMiterD,
          dR: reverseMiterD,
          joinUL: false,
          joinUR: false,
          joinDL: false,
          joinDR: false,
          situationIndex: 2,
        });
      }
      return JoinResult({
        uL: nUL,
        u: midL,
        uR: nUR,
        c: midL,
        dL: nDL,
        d: midL,
        dR: nDR,
        joinUL: false,
        joinUR: false,
        joinDL: false,
        joinDR: false,
        situationIndex: 3,
      });
    }

    if (sideUR === 1) {
      return JoinResult({
        uL: miterU.right,
        u: miterU.mid,
        uR: miterU.left,
        c: add(miterU.mid, normalize(sub(reverseMiterD, miterU.mid))),
        dL: reverseMiterD,
        d: reverseMiterD,
        dR: reverseMiterD,
        joinUL: true,
        joinUR: true,
        joinDL: false,
        joinDR: false,
        situationIndex: 4,
      });
    }

    return JoinResult({
      uL: reverseMiterU,
      u: reverseMiterU,
      uR: reverseMiterU,
      c: add(miterD.mid, normalize(sub(reverseMiterU, miterD.mid))),
      dL: miterD.left,
      d: miterD.mid,
      dR: miterD.right,
      joinUL: false,
      joinUR: false,
      joinDL: true,
      joinDR: true,
      situationIndex: 5,
    });
  },
);
