import tgpu from 'typegpu';
import { f32, struct, u32, vec2f } from 'typegpu/data';
import type { Infer, v2f } from 'typegpu/data';
import { dot, mul, normalize, select, sub } from 'typegpu/std';
import {
  addMul,
  bisectCcw,
  bisectNoCheck,
  midPoint,
  rot90ccw,
  rot90cw,
} from '../utils.ts';
import { externalNormals, limitTowardsMiddle, miterPoint } from './utils.ts';
import { roundJoin } from './joins/round.ts';
import { roundCap } from './caps/round.ts';
import { JoinPath } from './types.ts';
import { joinSituationIndex } from './joins/common.ts';

export const joinSlot = tgpu.slot(roundJoin);
export const startCapSlot = tgpu.slot(roundCap);
export const endCapSlot = tgpu.slot(roundCap);

const getJoinParent = tgpu.fn([u32], u32)((i) => (i - 4) >> 1);

const getJoinVertexPath = tgpu.fn([u32], JoinPath)((vertexIndex) => {
  let joinIndex = vertexIndex - 10;
  let depth = u32(0);
  let path = u32(0);
  while (joinIndex >= 4) {
    path = (path << 1) | (joinIndex & 1);
    joinIndex = getJoinParent(joinIndex);
    depth += 1;
  }
  return JoinPath({ joinIndex, path, depth });
});

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
  const joinPath = getJoinVertexPath(vertexIndex);

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

  const nBC = normalize(BC);

  let d0 = eBC.n1;
  let d4 = eBC.n2;
  let d5 = eBC.n2;
  let d9 = eBC.n1;

  const situationIndexB = joinSituationIndex(eAB.n1, eBC.n1, eAB.n2, eBC.n2);
  const situationIndexC = joinSituationIndex(eBC.n1, eCD.n1, eBC.n2, eCD.n2);
  let joinBu = true;
  let joinBd = true;
  let joinCu = true;
  let joinCd = true;
  if (!isCapB) {
    if (situationIndexB === 1 || situationIndexB === 5) {
      d4 = miterPoint(eBC.n2, eAB.n2);
      joinBd = false;
    }
    if (situationIndexB === 4 || situationIndexB === 5) {
      d0 = miterPoint(eAB.n1, eBC.n1);
      joinBu = false;
    }
  }
  if (!isCapC) {
    if (situationIndexC === 1 || situationIndexC === 5) {
      d5 = miterPoint(eCD.n2, eBC.n2);
      joinCd = false;
    }
    if (situationIndexC === 4 || situationIndexC === 5) {
      d9 = miterPoint(eBC.n1, eCD.n1);
      joinCu = false;
    }
  }

  let v0 = addMul(B.position, d0, B.radius);
  let v4 = addMul(B.position, d4, B.radius);
  let v5 = addMul(C.position, d5, C.radius);
  let v9 = addMul(C.position, d9, C.radius);

  const midBC = midPoint(B.position, C.position);
  const tBC1 = rot90cw(eBC.n1);
  const tBC2 = rot90ccw(eBC.n2);

  const lim16 = limitTowardsMiddle(midBC, tBC1, v0, v9);
  const lim38 = limitTowardsMiddle(midBC, tBC2, v4, v5);
  v0 = lim16.a;
  v9 = lim16.b;
  v4 = lim38.a;
  v5 = lim38.b;

  // after this point we need to process only one of the joins!
  const isCSide = (vertexIndex < 10 && vertexIndex >= 5) ||
    vertexIndex >= 10 && joinPath.joinIndex >= 2;

  let V = B;
  let isCap = isCapB;
  let cap = startCapSlot.$(eBC.n1, mul(nBC, -1), eBC.n2);
  let j1 = eAB.n1;
  let j2 = eBC.n1;
  let j3 = eAB.n2;
  let j4 = eBC.n2;
  let vu = v0;
  let vd = v4;
  let joinU = joinBu;
  let joinD = joinBd;
  if (isCSide) {
    V = C;
    isCap = isCapC;
    cap = endCapSlot.$(eBC.n2, nBC, eBC.n1);
    j4 = eBC.n1;
    j3 = eCD.n1;
    j2 = eBC.n2;
    j1 = eCD.n2;
    vu = v5;
    vd = v9;
    joinU = joinCd;
    joinD = joinCu;
  }

  const join = joinSlot.$(j1, j2, j3, j4);
  if (isCap) {
    join.uR = cap.right;
    join.u = cap.rightForward;
    join.c = cap.forward;
    join.d = cap.leftForward;
    join.dR = cap.left;
  }

  if (vertexIndex >= 10) {
    const shouldJoin = [
      u32(joinBu),
      u32(joinBd),
      u32(joinCd),
      u32(joinCu),
    ];

    const joinIndex = joinPath.joinIndex;
    if (shouldJoin[joinPath.joinIndex] === 0) {
      const noJoinPoints = [v0, v4, v5, v9];
      return {
        situationIndex: join.situationIndex,
        vertexPosition: noJoinPoints[joinPath.joinIndex] as v2f,
      };
    }

    // deno-fmt-ignore
    const parents = [
      join.uR, join.u, join.d, join.dR,
      join.uR, join.u, join.d, join.dR,
    ];

    let d0 = parents[joinIndex * 2] as v2f;
    let d1 = parents[joinIndex * 2 + 1] as v2f;
    let d = bisectCcw(d0, d1);
    let path = joinPath.path;
    for (let depth = joinPath.depth; depth > 0; depth -= 1) {
      const isLeftChild = (path & 1) === 0;
      d0 = select(d, d0, isLeftChild);
      d1 = select(d1, d, isLeftChild);
      d = bisectNoCheck(d0, d1);
      path >>= 1;
    }
    return {
      situationIndex: join.situationIndex,
      vertexPosition: addMul(V.position, d, V.radius),
    };
  }

  const removeCenter = !isCap && join.situationIndex === 5;
  const v1 = select(vu, addMul(V.position, join.u, V.radius), joinU);
  const v2 = select(addMul(V.position, join.c, V.radius), vu, removeCenter);
  const v3 = select(vd, addMul(V.position, join.d, V.radius), joinD);
  const points = [vu, v1, v2, v3, vd];
  return {
    vertexPosition: points[vertexIndex % 5] as v2f,
    situationIndex: join.situationIndex,
  };
});
