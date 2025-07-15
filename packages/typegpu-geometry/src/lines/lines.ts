import tgpu from 'typegpu';
import { bool, f32, struct, u32, vec2f } from 'typegpu/data';
import type { v2f } from 'typegpu/data';
import { add, dot, mul, normalize, select, sub } from 'typegpu/std';
import {
  addMul,
  bisectCcw,
  bisectNoCheck,
  cross2d,
  midPoint,
  rot90ccw,
  rot90cw,
} from '../utils.ts';
import { externalNormals, limitTowardsMiddle, miterPoint } from './utils.ts';

const JOIN_LIMIT = tgpu['~unstable'].const(f32, 0.999);

const JoinResult = struct({
  uL: vec2f,
  u: vec2f,
  uR: vec2f,
  c: vec2f,
  dL: vec2f,
  d: vec2f,
  dR: vec2f,
  situationIndex: u32,
  joinUL: bool,
  joinDL: bool,
  joinUR: bool,
  joinDR: bool,
});

const solveRoundJoin = tgpu.fn(
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
    const center = mul(add(add(nUL, nUR), add(nDL, nDR)), 0.25);

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

    if (sideUR === sideDR) {
      const side = sideUR;
      const XUR = select(xUR, side * 2 - xUR, yUR < 0);
      const XDR = select(xDR, side * 2 - xDR, yDR < 0);
      const clockWise = (side * (XUR - XDR)) <= 0;
      if (side === 1) {
        if (clockWise) {
          return JoinResult({
            uL: select(midpU, nUL, joinU),
            u: select(midpU, midU, joinU),
            uR: select(midpU, nUR, joinU),
            c: center,
            dL: select(midpD, nDL, joinD),
            d: select(midpD, midD, joinD),
            dR: select(midpD, nDR, joinD),
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
        uL: select(midpU, nUL, joinU),
        u: select(midpU, midU, joinU),
        uR: select(midpU, nUR, joinU),
        c: center,
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
      c: center,
      dL: select(midpD, nDL, joinD),
      d: select(midpD, midD, joinD),
      dR: select(midpD, nDR, joinD),
      joinUL: false,
      joinUR: false,
      joinDL: joinD,
      joinDR: joinD,
      situationIndex: 5,
    });
  },
);

const solveMiterJoin = tgpu.fn(
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
    const center = mul(add(add(nUL, nUR), add(nDL, nDR)), 0.25);

    const midR = bisectCcw(nUR, nDR);
    const midL = bisectCcw(nDL, nUL);
    const reverseMiterU = miterPoint(nUL, nUR);
    const reverseMiterD = miterPoint(nDR, nDL);

    const miterU = miterPoint(nUR, nUL);
    const miterD = miterPoint(nDL, nDR);

    const joinU = dot(nUL, nUR) < JOIN_LIMIT.$;
    const joinD = dot(nDL, nDR) < JOIN_LIMIT.$;

    if (sideUR === sideDR) {
      const side = sideUR;
      const XUR = select(xUR, side * 2 - xUR, yUR < 0);
      const XDR = select(xDR, side * 2 - xDR, yDR < 0);
      const clockWise = (side * (XUR - XDR)) <= 0;
      if (side === 1) {
        if (clockWise) {
          return JoinResult({
            uL: miterU,
            u: miterU,
            uR: miterU,
            c: center,
            dL: miterD,
            d: miterD,
            dR: miterD,
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
        uL: miterU,
        u: miterU,
        uR: miterU,
        c: center,
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
      c: center,
      dL: miterD,
      d: miterD,
      dR: miterD,
      joinUL: false,
      joinUR: false,
      joinDL: joinD,
      joinDR: joinD,
      situationIndex: 5,
    });
  },
);

const solveCap = tgpu.fn(
  [vec2f, vec2f],
  JoinResult,
)(
  (a, b) => {
    const mid = bisectCcw(a, b);
    return JoinResult({
      uL: b,
      u: mid,
      uR: a,
      c: mid,
      dL: a,
      d: mid,
      dR: b,
      joinUL: true,
      joinUR: true,
      joinDL: true,
      joinDR: true,
      situationIndex: 0,
    });
  },
);

const LineSegmentOutput = struct({
  vertexPosition: vec2f,
  situationIndex: u32,
});

export const LineSegmentVertex = struct({
  position: vec2f,
  radius: f32,
});

export const lineSegmentVariableWidth = tgpu.fn([
  u32,
  LineSegmentVertex,
  LineSegmentVertex,
  LineSegmentVertex,
  LineSegmentVertex,
], LineSegmentOutput)((vertexIndex, A, B, C, D) => {
  const AB = sub(B.position, A.position);
  const BC = sub(C.position, B.position);
  const CD = sub(D.position, C.position);

  const radiusABDelta = A.radius - B.radius;
  const radiusBCDelta = B.radius - C.radius;
  const radiusCDDelta = C.radius - D.radius;

  // segments where one end completely contains the other are skipped
  // TODO: we should probably render a circle in case one of the ends is a cap
  if (dot(BC, BC) <= radiusBCDelta * radiusBCDelta) {
    return {
      vertexPosition: vec2f(0, 0),
      uv: vec2f(0, 0),
      situationIndex: 0,
    };
  }

  const isCapB = dot(AB, AB) <= radiusABDelta * radiusABDelta;
  const isCapC = dot(CD, CD) <= radiusCDDelta * radiusCDDelta;

  const eAB = externalNormals(AB, A.radius, B.radius);
  const eBC = externalNormals(BC, B.radius, C.radius);
  const eCD = externalNormals(CD, C.radius, D.radius);

  const nAB = normalize(AB);
  const nBC = normalize(BC);

  let joinB = solveCap(eBC.n1, eBC.n2);
  if (!isCapB) {
    joinB = solveRoundJoin(nAB, eAB.n1, eBC.n1, eAB.n2, eBC.n2);
  }

  let v0 = addMul(B.position, joinB.u, B.radius);
  let v1 = addMul(B.position, joinB.uR, B.radius);
  let v2 = addMul(B.position, joinB.c, B.radius);
  let v3 = addMul(B.position, joinB.dR, B.radius);
  let v4 = addMul(B.position, joinB.d, B.radius);

  let joinC = solveCap(eBC.n2, eBC.n1);
  if (!isCapC) {
    joinC = solveRoundJoin(nBC, eBC.n1, eCD.n1, eBC.n2, eCD.n2);
  }

  let v5 = addMul(C.position, joinC.u, C.radius);
  let v6 = addMul(C.position, joinC.uL, C.radius);
  let v7 = addMul(C.position, joinC.c, C.radius);
  let v8 = addMul(C.position, joinC.dL, C.radius);
  let v9 = addMul(C.position, joinC.d, C.radius);

  const midBC = midPoint(B.position, C.position);
  const tBC1 = rot90cw(eBC.n1);
  const tBC2 = rot90ccw(eBC.n2);

  const lim16 = limitTowardsMiddle(midBC, tBC1, v1, v6);
  const lim38 = limitTowardsMiddle(midBC, tBC2, v3, v8);
  v1 = lim16.a;
  v6 = lim16.b;
  v3 = lim38.a;
  v8 = lim38.b;

  if (!joinB.joinUR) {
    v0 = v1;
  }
  if (!joinB.joinDR) {
    v4 = v3;
  }
  if (!joinC.joinUL) {
    v5 = v6;
  }
  if (!joinC.joinDL) {
    v9 = v8;
  }
  if (joinB.situationIndex === 2) {
    // remove central triangle but only after limits are applied
    v2 = v1;
  }
  if (joinC.situationIndex === 2) {
    // remove central triangle but only after limits are applied
    v7 = v6;
  }

  let d10 = joinB.u;
  let d11 = joinB.d;
  let d12 = joinC.u;
  let d13 = joinC.d;
  let d14 = joinB.u;
  let d15 = joinB.u;
  let d16 = joinB.d;
  let d17 = joinB.d;
  let d18 = joinC.u;
  let d19 = joinC.u;
  let d20 = joinC.d;
  let d21 = joinC.d;

  if (joinB.joinUR) {
    d10 = bisectCcw(joinB.uR, joinB.u);
    d14 = bisectNoCheck(d10, joinB.u);
    d15 = bisectNoCheck(joinB.uR, d10);
  }
  if (joinB.joinDR) {
    d11 = bisectCcw(joinB.d, joinB.dR);
    d16 = bisectNoCheck(d11, joinB.dR);
    d17 = bisectNoCheck(joinB.d, d11);
  }
  if (joinC.joinUL) {
    d12 = bisectCcw(joinC.u, joinC.uL);
    d18 = bisectNoCheck(joinC.u, d12);
    d19 = bisectNoCheck(d12, joinC.uL);
  }
  if (joinC.joinDL) {
    d13 = bisectCcw(joinC.dL, joinC.d);
    d20 = bisectNoCheck(joinC.dL, d13);
    d21 = bisectNoCheck(d13, joinC.d);
  }

  const v10 = addMul(B.position, d10, B.radius);
  const v11 = addMul(B.position, d11, B.radius);
  const v12 = addMul(C.position, d12, C.radius);
  const v13 = addMul(C.position, d13, C.radius);
  const v14 = addMul(B.position, d14, B.radius);
  const v15 = addMul(B.position, d15, B.radius);
  const v16 = addMul(B.position, d16, B.radius);
  const v17 = addMul(B.position, d17, B.radius);
  const v18 = addMul(C.position, d18, C.radius);
  const v19 = addMul(C.position, d19, C.radius);
  const v20 = addMul(C.position, d20, C.radius);
  const v21 = addMul(C.position, d21, C.radius);

  // deno-fmt-ignore
  const points = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21];

  return {
    vertexPosition: points[vertexIndex] as v2f,
    situationIndex: joinB.situationIndex,
  };
});

export const lineSingleSegmentVariableWidth = tgpu.fn([
  u32,
  LineSegmentVertex,
  LineSegmentVertex,
], LineSegmentOutput)((vertexIndex, A, B) => {
  const AB = sub(B.position, A.position);
  const eAB = externalNormals(AB, A.radius, B.radius);

  const joinA = solveCap(eAB.n1, eAB.n2);
  const joinB = solveCap(eAB.n2, eAB.n1);

  const v0 = addMul(A.position, joinA.u, A.radius);
  const v1 = addMul(A.position, joinA.uR, A.radius);
  const v2 = addMul(A.position, joinA.c, A.radius);
  const v3 = addMul(A.position, joinA.dR, A.radius);
  const v4 = addMul(A.position, joinA.d, A.radius);
  const v5 = addMul(B.position, joinB.u, B.radius);
  const v6 = addMul(B.position, joinB.uL, B.radius);
  const v7 = addMul(B.position, joinB.c, B.radius);
  const v8 = addMul(B.position, joinB.dL, B.radius);
  const v9 = addMul(B.position, joinB.d, B.radius);

  let d10 = joinA.u;
  let d11 = joinA.d;
  let d12 = joinA.u;
  let d13 = joinA.d;
  let d14 = joinA.u;
  let d15 = joinA.u;
  let d16 = joinB.d;
  let d17 = joinB.d;
  let d18 = joinB.u;
  let d19 = joinB.u;
  let d20 = joinB.d;
  let d21 = joinB.d;

  if (joinA.joinUR) {
    d10 = bisectCcw(joinA.uR, joinA.u);
    d14 = bisectNoCheck(d10, joinA.u);
    d15 = bisectNoCheck(joinA.uR, d10);
  }
  if (joinA.joinDR) {
    d11 = bisectCcw(joinA.d, joinA.dR);
    d16 = bisectNoCheck(d11, joinA.dR);
    d17 = bisectNoCheck(joinA.d, d11);
  }
  if (joinB.joinUL) {
    d12 = bisectCcw(joinB.u, joinB.uL);
    d18 = bisectNoCheck(joinB.u, d12);
    d19 = bisectNoCheck(d12, joinB.uL);
  }
  if (joinB.joinDL) {
    d13 = bisectCcw(joinB.dL, joinB.d);
    d20 = bisectNoCheck(joinB.dL, d13);
    d21 = bisectNoCheck(d13, joinB.d);
  }

  const v10 = addMul(A.position, d10, A.radius);
  const v11 = addMul(A.position, d11, A.radius);
  const v12 = addMul(B.position, d12, B.radius);
  const v13 = addMul(B.position, d13, B.radius);
  const v14 = addMul(A.position, d14, A.radius);
  const v15 = addMul(A.position, d15, A.radius);
  const v16 = addMul(A.position, d16, A.radius);
  const v17 = addMul(A.position, d17, A.radius);
  const v18 = addMul(B.position, d18, B.radius);
  const v19 = addMul(B.position, d19, B.radius);
  const v20 = addMul(B.position, d20, B.radius);
  const v21 = addMul(B.position, d21, B.radius);

  // deno-fmt-ignore
  const points = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21];

  return {
    vertexPosition: points[vertexIndex] as v2f,
    situationIndex: joinA.situationIndex,
  };
});
