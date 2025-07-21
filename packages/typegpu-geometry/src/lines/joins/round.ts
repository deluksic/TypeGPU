import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { add, dot, mul, select } from 'typegpu/std';
import { bisectCcw } from '../../utils.ts';
import { JoinResult } from '../types.ts';
import { intersectLines, miterPoint, miterPointNoCheck } from '../utils.ts';
import { JOIN_LIMIT } from '../constants.ts';
import { joinSituationIndex } from './common.ts';

export const roundJoin = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f],
  JoinResult,
)((ul, ur, dl, dr) => {
  const situationIndex = joinSituationIndex(ul, ur, dl, dr);

  const joinU = dot(ul, ur) < JOIN_LIMIT.$;
  const joinD = dot(dl, dr) < JOIN_LIMIT.$;

  const midU = bisectCcw(ur, ul);
  const midD = bisectCcw(dl, dr);
  const midR = bisectCcw(ur, dr);
  const midL = bisectCcw(dl, ul);

  // these need to be computed separately, because in
  // one case we need CW miter and the other is unstable at 180 deg
  // probably can be fixed to compute only once!
  const miterU = miterPointNoCheck(ul, ur);
  const miterD = miterPointNoCheck(dr, dl);
  const reverseMiterU = select(miterU, miterPoint(ul, ur), joinU);
  const reverseMiterD = select(miterD, miterPoint(dr, dl), joinD);

  const maybeJoinedUL = select(miterU, ul, joinU);
  const maybeJoinedU = select(miterU, midU, joinU);
  const maybeJoinedUR = select(miterU, ur, joinU);
  const maybeJoinedDL = select(miterD, dl, joinD);
  const maybeJoinedD = select(miterD, midD, joinD);
  const maybeJoinedDR = select(miterD, dr, joinD);

  const crossCenter = intersectLines(ul, dl, ur, dr).point;
  const averageCenter = mul(
    add(
      add(maybeJoinedUR, maybeJoinedUL),
      add(maybeJoinedDL, maybeJoinedDR),
    ),
    0.25,
  );

  if (situationIndex === 0) {
    return JoinResult({
      uL: maybeJoinedUL,
      u: maybeJoinedU,
      uR: maybeJoinedUR,
      c: averageCenter,
      dL: maybeJoinedDL,
      d: maybeJoinedD,
      dR: maybeJoinedDR,
      situationIndex,
    });
  }

  if (situationIndex === 1) {
    return JoinResult({
      uL: maybeJoinedUL,
      u: maybeJoinedU,
      uR: maybeJoinedUR,
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
      dL: maybeJoinedDL,
      d: maybeJoinedD,
      dR: maybeJoinedDR,
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
