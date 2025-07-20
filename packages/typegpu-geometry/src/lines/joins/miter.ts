import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { add, dot, mul, normalize, select } from 'typegpu/std';
import { bisectCcw, cross2d } from '../../utils.ts';
import { JoinResult } from '../types.ts';
import {
  intersectLines,
  isCCW,
  miterLimit,
  miterPoint,
  miterPointNoCheck,
  rank3,
} from '../utils.ts';
import { JOIN_LIMIT } from '../constants.ts';

export const miterJoin = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f],
  JoinResult,
)(
  (ul, ur, dl, dr) => {
    // ur is the reference vector
    // we find all 6 orderings of the remaining ul, dl, dr
    const crossUL = cross2d(ur, ul);
    const crossDL = cross2d(ur, dl);
    const crossDR = cross2d(ur, dr);
    const signUL = crossUL >= 0;
    const signDL = crossDL >= 0;
    const signDR = crossDR >= 0;
    const dotUL = dot(ur, ul);
    const dotDL = dot(ur, dl);
    const dotDR = dot(ur, dr);

    const situationIndex = rank3(
      isCCW(dotUL, signUL, dotDL, signDL),
      isCCW(dotDL, signDL, dotDR, signDR),
      isCCW(dotUL, signUL, dotDR, signDR),
    );

    const midR = bisectCcw(ur, dr);
    const midL = bisectCcw(dl, ul);

    const miterU = miterLimit(ur, ul);
    const miterD = miterLimit(dl, dr);

    const joinU = dot(ul, ur) < JOIN_LIMIT.$;
    const joinD = dot(dl, dr) < JOIN_LIMIT.$;
    const reverseMiterU = select(
      miterPointNoCheck(ul, ur),
      miterPoint(ul, ur),
      joinU,
    );
    const reverseMiterD = select(
      miterPointNoCheck(dr, dl),
      miterPoint(dr, dl),
      joinD,
    );

    const crossCenter = intersectLines(ul, dl, ur, dr).point;
    const averageCenter = mul(
      add(
        normalize(miterU.mid),
        normalize(miterD.mid),
      ),
      0.5,
    );

    if (situationIndex === 0) {
      return JoinResult({
        uL: ul,
        u: miterU.mid,
        uR: ur,
        c: averageCenter,
        dL: dl,
        d: miterD.mid,
        dR: dr,
        joinU: true,
        joinD: true,
        situationIndex,
      });
    }

    if (situationIndex === 1) {
      return JoinResult({
        uL: ul,
        u: miterU.mid,
        uR: ur,
        c: crossCenter,
        dL: reverseMiterD,
        d: reverseMiterD,
        dR: reverseMiterD,
        joinU: true,
        joinD: false,
        situationIndex,
      });
    }

    if (situationIndex === 2) {
      return JoinResult({
        uL: ul,
        u: midR,
        uR: ur,
        c: midR,
        dL: dl,
        d: midR,
        dR: dr,
        joinU: true,
        joinD: true,
        situationIndex,
      });
    }

    if (situationIndex === 3) {
      return JoinResult({
        uL: ul,
        u: midL,
        uR: ur,
        c: midL,
        dL: dl,
        d: midL,
        dR: dr,
        joinU: true,
        joinD: true,
        situationIndex,
      });
    }

    if (situationIndex === 4) {
      return JoinResult({
        uL: reverseMiterU,
        u: reverseMiterU,
        uR: reverseMiterU,
        c: crossCenter,
        dL: dl,
        d: miterD.mid,
        dR: dr,
        joinU: false,
        joinD: true,
        situationIndex,
      });
    }

    // situationIndex === 5
    return JoinResult({
      uL: reverseMiterU,
      u: reverseMiterU,
      uR: reverseMiterU,
      c: averageCenter,
      dL: reverseMiterD,
      d: reverseMiterD,
      dR: reverseMiterD,
      joinU: false,
      joinD: false,
      situationIndex,
    });
  },
);
