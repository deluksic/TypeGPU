import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { add, dot, mul, normalize, select } from 'typegpu/std';
import { bisectCcw } from '../../utils.ts';
import { JoinResult } from '../types.ts';
import {
  intersectLines,
  miterLimit,
  miterPoint,
  miterPointNoCheck,
} from '../utils.ts';
import { JOIN_LIMIT } from '../constants.ts';
import { joinSituationIndex } from './common.ts';

export const miterJoin = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f],
  JoinResult,
)((ul, ur, dl, dr) => {
  const situationIndex = joinSituationIndex(ul, ur, dl, dr);

  const joinU = dot(ul, ur) < JOIN_LIMIT.$;
  const joinD = dot(dl, dr) < JOIN_LIMIT.$;

  const miterU = miterLimit(ur, ul);
  const miterD = miterLimit(dl, dr);
  const midR = bisectCcw(ur, dr);
  const midL = bisectCcw(dl, ul);

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
    situationIndex,
  });
});
