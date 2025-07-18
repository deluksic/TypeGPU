import tgpu from 'typegpu';
import { bool, f32, struct, u32, vec2f } from 'typegpu/data';
import type { Infer, v2f } from 'typegpu/data';
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
import {
  externalNormals,
  intersectLines,
  limitTowardsMiddle,
  miterLimit,
  miterPoint,
} from './utils.ts';

const JOIN_LIMIT = tgpu['~unstable'].const(f32, 0.999);

type JoinResult = Infer<typeof JoinResult>;
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

const getJoinParent = tgpu.fn([u32], u32)((i) => (i - 4) >> 1);

const LineSegmentOutput = struct({
  vertexPosition: vec2f,
  situationIndex: u32,
});

type LineSegmentVertex = Infer<typeof LineSegmentVertex>;
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
  // TODO: we should probably render a circle in some cases
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

  const capB = solveCap(eBC.n1, eBC.n2);
  let joinB = solveRoundJoin(nAB, eAB.n1, eBC.n1, eAB.n2, eBC.n2);
  if (isCapB) {
    joinB = capB;
  }

  let v0 = addMul(B.position, joinB.uR, B.radius);
  let v1 = addMul(B.position, joinB.u, B.radius);
  let v2 = addMul(B.position, joinB.c, B.radius);
  let v3 = addMul(B.position, joinB.d, B.radius);
  let v4 = addMul(B.position, joinB.dR, B.radius);

  const capC = solveCap(eBC.n2, eBC.n1);
  let joinC = solveRoundJoin(nBC, eBC.n1, eCD.n1, eBC.n2, eCD.n2);
  if (isCapC) {
    joinC = capC;
  }

  let v5 = addMul(C.position, joinC.dL, C.radius);
  let v6 = addMul(C.position, joinC.d, C.radius);
  let v7 = addMul(C.position, joinC.c, C.radius);
  let v8 = addMul(C.position, joinC.u, C.radius);
  let v9 = addMul(C.position, joinC.uL, C.radius);

  const midBC = midPoint(B.position, C.position);
  const tBC1 = rot90cw(eBC.n1);
  const tBC2 = rot90ccw(eBC.n2);

  const lim16 = limitTowardsMiddle(midBC, tBC1, v0, v9);
  const lim38 = limitTowardsMiddle(midBC, tBC2, v4, v5);
  v0 = lim16.a;
  v9 = lim16.b;
  v4 = lim38.a;
  v5 = lim38.b;

  if (!joinB.joinUR) {
    v1 = v0;
  }
  if (!joinB.joinDR) {
    v3 = v4;
  }
  if (!joinC.joinUL) {
    v8 = v9;
  }
  if (!joinC.joinDL) {
    v6 = v5;
  }
  if (joinB.situationIndex === 2) {
    // remove central triangle but only after limits are applied
    v2 = v0;
  }
  if (joinC.situationIndex === 2) {
    // remove central triangle but only after limits are applied
    v7 = v9;
  }

  const points = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9];

  if (vertexIndex < 10) {
    return {
      vertexPosition: points[vertexIndex] as v2f,
      situationIndex: joinB.situationIndex,
    };
  }

  const shouldJoin = [
    u32(joinB.joinUR),
    u32(joinB.joinDR),
    u32(joinC.joinDL),
    u32(joinC.joinUL),
  ];

  // deno-fmt-ignore
  const parents = [
    joinB.uR, joinB.u,
    joinB.d, joinB.dR,
    joinC.dL, joinC.d,
    joinC.u, joinC.uL,
  ];

  const noJoinPoints = [v0, v4, v5, v9];

  let i = vertexIndex - 10;
  let depth = u32(0);
  let path = u32(0);
  while (i >= 4) {
    path = (path << 1) | (i & 1);
    i = getJoinParent(i);
    depth += 1;
  }
  let lineVertex = B;
  if (i >= 2) {
    lineVertex = C;
  }
  if (shouldJoin[i] === 0) {
    return {
      situationIndex: joinB.situationIndex,
      vertexPosition: noJoinPoints[i] as v2f,
    };
  }
  let d0 = parents[i * 2] as v2f;
  let d1 = parents[i * 2 + 1] as v2f;
  let d = bisectCcw(d0, d1);
  while (depth > 0) {
    const isLeftChild = (path & 1) === 0;
    path = path >> 1;
    depth -= 1;
    d0 = select(d, d0, isLeftChild);
    d1 = select(d1, d, isLeftChild);
    d = bisectNoCheck(d0, d1);
  }
  return {
    situationIndex: joinB.situationIndex,
    vertexPosition: addMul(lineVertex.position, d, lineVertex.radius),
  };
});
