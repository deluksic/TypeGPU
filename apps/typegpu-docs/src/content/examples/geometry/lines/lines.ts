import tgpu from 'typegpu';
import { struct, u32, vec2f } from 'typegpu/data';
import { add, dot, mul, select, sign } from 'typegpu/std';
import { cross2d, intersectLines, midDirection, miterPoint } from './utils.ts';

export const JoinResult = struct({
  uL: vec2f,
  u: vec2f,
  uR: vec2f,
  c: vec2f,
  dL: vec2f,
  d: vec2f,
  dR: vec2f,
  situationIndex: u32,
});

export const solveJoin = tgpu.fn(
  [vec2f, vec2f, vec2f, vec2f, vec2f],
  JoinResult,
)(
  (nL, nUL, nUR, nDL, nDR) => {
    const xL = dot(nL, nUL); // == dot(nL, nDL)
    const xUR = dot(nL, nUR);
    const xDR = dot(nL, nDR);
    const yUR = cross2d(nL, nUR);
    const yDR = cross2d(nL, nDR);
    const sideUR = sign(xUR - xL);
    const sideDR = sign(xDR - xL);
    const int = intersectLines(nUL, nDL, nUR, nDR);
    let center = mul(add(add(nUL, nUR), add(nDL, nDR)), 0.25);
    if (int.valid && int.t >= 0 && int.t <= 1) {
      center = int.point;
    }

    const midU = midDirection(nUL, nUR);
    const midD = midDirection(nDR, nDL);
    const midR = midDirection(nDR, nUR);
    const miterU = miterPoint(nUL, nUR);
    const miterD = miterPoint(nDR, nDL);

    if (sideUR === sideDR) {
      const side = sideUR;
      const XUR = select(xUR, side * 2 - xUR, yUR < 0);
      const XDR = select(xDR, side * 2 - xDR, yDR < 0);
      const clockWise = (side * (XUR - XDR)) <= 0;
      if (side >= 0) {
        if (clockWise) {
          return JoinResult({
            uL: nUL,
            u: midU,
            uR: nUR,
            c: center,
            dL: nDL,
            d: midD,
            dR: nDR,
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
          situationIndex: 1,
        });
      }
      // side == -1
      if (clockWise) {
        return JoinResult({
          uL: miterU,
          u: miterU,
          uR: miterU,
          c: miterU, // remove inner triangle
          dL: miterD,
          d: miterD,
          dR: miterD,
          situationIndex: 2,
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
        situationIndex: 3,
      });
    }

    if (sideUR >= 0) {
      return JoinResult({
        uL: nUL,
        u: midU,
        uR: nUR,
        c: center,
        dL: miterD,
        d: miterD,
        dR: miterD,
        situationIndex: 4,
      });
    }

    return JoinResult({
      uL: miterU,
      u: miterU,
      uR: miterU,
      c: center,
      dL: nDL,
      d: midD,
      dR: nDR,
      situationIndex: 5,
    });
  },
);

export const solveCap = tgpu.fn(
  [vec2f, vec2f],
  JoinResult,
)(
  (a, b) => {
    const mid = midDirection(b, a);
    return JoinResult({
      uL: b,
      u: mid,
      uR: a,
      c: mid,
      dL: a,
      d: mid,
      dR: b,
      situationIndex: 0,
    });
  },
);
