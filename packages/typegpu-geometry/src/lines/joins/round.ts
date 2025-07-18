import tgpu from 'typegpu';
import { f32, vec2f } from 'typegpu/data';
import { add, dot, mul, select } from 'typegpu/std';
import { bisectCcw, cross2d, midPoint } from '../../utils.ts';
import { JoinResult } from '../types.ts';
import { intersectLines, miterPoint } from '../utils.ts';
import { JOIN_LIMIT } from '../constants.ts';

export const roundJoin = tgpu.fn(
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

    const midU = bisectCcw(nUR, nUL);
    const midD = bisectCcw(nDL, nDR);
    const midR = bisectCcw(nUR, nDR);
    const midL = bisectCcw(nDL, nUL);
    const reverseMiterU = miterPoint(nUL, nUR);
    const reverseMiterD = miterPoint(nDR, nDL);

    const joinU = dot(nUL, nUR) < JOIN_LIMIT.$;
    const joinD = dot(nDL, nDR) < JOIN_LIMIT.$;
    const midpU = midPoint(nUL, nUR);
    const midpD = midPoint(nDL, nDR);

    const maybeJoinedUL = select(midpU, nUL, joinU);
    const maybeJoinedU = select(midpU, midU, joinU);
    const maybeJoinedUR = select(midpU, nUR, joinU);
    const maybeJoinedDL = select(midpD, nDL, joinD);
    const maybeJoinedD = select(midpD, midD, joinD);
    const maybeJoinedDR = select(midpD, nDR, joinD);

    const center = mul(
      add(
        add(
          maybeJoinedUR,
          maybeJoinedUL,
        ),
        add(
          maybeJoinedDL,
          maybeJoinedDR,
        ),
      ),
      0.25,
    );

    const crossCenter = intersectLines(nUL, nDL, nUR, nDR).point;

    if (sideUR === sideDR) {
      const side = sideUR;
      const XUR = select(xUR, side * 2 - xUR, yUR < 0);
      const XDR = select(xDR, side * 2 - xDR, yDR < 0);
      const clockWise = (side * (XUR - XDR)) <= 0;
      if (side === 1) {
        if (clockWise) {
          return JoinResult({
            uL: maybeJoinedUL,
            u: maybeJoinedU,
            uR: maybeJoinedUR,
            c: center,
            dL: maybeJoinedDL,
            d: maybeJoinedD,
            dR: maybeJoinedDR,
            joinUL: joinU,
            joinUR: joinU,
            joinDL: joinD,
            joinDR: joinD,
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
          joinUL: true,
          joinUR: true,
          joinDL: true,
          joinDR: true,
          situationIndex: 1,
        });
      }
      // side == -1
      if (clockWise) {
        return JoinResult({
          uL: reverseMiterU,
          u: reverseMiterU,
          uR: reverseMiterU,
          c: center,
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
        joinUL: true,
        joinUR: true,
        joinDL: true,
        joinDR: true,
        situationIndex: 3,
      });
    }

    if (sideUR === 1) {
      return JoinResult({
        uL: maybeJoinedUL,
        u: maybeJoinedU,
        uR: maybeJoinedUR,
        c: crossCenter,
        dL: reverseMiterD,
        d: reverseMiterD,
        dR: reverseMiterD,
        joinUL: joinU,
        joinUR: joinU,
        joinDL: false,
        joinDR: false,
        situationIndex: 4,
      });
    }

    return JoinResult({
      uL: reverseMiterU,
      u: reverseMiterU,
      uR: reverseMiterU,
      c: crossCenter,
      dL: maybeJoinedDL,
      d: maybeJoinedD,
      dR: maybeJoinedDR,
      joinUL: false,
      joinUR: false,
      joinDL: joinD,
      joinDR: joinD,
      situationIndex: 5,
    });
  },
);
