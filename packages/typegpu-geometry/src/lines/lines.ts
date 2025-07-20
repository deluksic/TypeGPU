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
import {
  externalNormals,
  limitTowardsMiddle,
  miterPoint,
  miterPointNoCheck,
} from './utils.ts';
import { roundJoin } from './joins/round.ts';
import { roundCap } from './caps/round.ts';
import { JoinPath } from './types.ts';

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

  const nAB = normalize(AB);
  const nBC = normalize(BC);

  const capB = startCapSlot.$(eBC.n1, mul(nBC, -1), eBC.n2);
  const joinB = joinSlot.$(eAB.n1, eBC.n1, eAB.n2, eBC.n2);
  if (isCapB) {
    joinB.uR = capB.right;
    joinB.u = capB.rightForward;
    joinB.c = capB.forward;
    joinB.d = capB.leftForward;
    joinB.dR = capB.left;
    joinB.joinU = capB.joinRight;
    joinB.joinD = capB.joinLeft;
  }

  let v0 = addMul(B.position, joinB.uR, B.radius);
  let v1 = addMul(B.position, joinB.u, B.radius);
  let v2 = addMul(B.position, joinB.c, B.radius);
  let v3 = addMul(B.position, joinB.d, B.radius);
  let v4 = addMul(B.position, joinB.dR, B.radius);

  const capC = endCapSlot.$(eBC.n2, nBC, eBC.n1);
  const joinC = joinSlot.$(eBC.n1, eCD.n1, eBC.n2, eCD.n2);
  if (isCapC) {
    joinC.dL = capC.right;
    joinC.d = capC.rightForward;
    joinC.c = capC.forward;
    joinC.u = capC.leftForward;
    joinC.uL = capC.left;
    joinC.joinD = capC.joinRight;
    joinC.joinU = capC.joinLeft;
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

  // if not a join, these need to be merged after limits are applied
  // in order for joins not to go into infinity
  if (!joinB.joinU) {
    v1 = v0;
  }
  if (!joinB.joinD) {
    v3 = v4;
  }
  if (!joinC.joinD) {
    v6 = v5;
  }
  if (!joinC.joinU) {
    v8 = v9;
  }

  // remove central triangle but only after limits are applied
  if (!isCapB && joinB.situationIndex === 5) {
    v2 = v0;
  }
  if (!isCapC && joinC.situationIndex === 5) {
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
    u32(joinB.joinU),
    u32(joinB.joinD),
    u32(joinC.joinD),
    u32(joinC.joinU),
  ];

  const noJoinPoints = [v0, v4, v5, v9];

  const joinIndex = joinPath.joinIndex;
  if (shouldJoin[joinPath.joinIndex] === 0) {
    return {
      situationIndex: joinB.situationIndex,
      vertexPosition: noJoinPoints[joinPath.joinIndex] as v2f,
    };
  }

  // deno-fmt-ignore
  const parents = [
    joinB.uR, joinB.u,
    joinB.d, joinB.dR,
    joinC.dL, joinC.d,
    joinC.u, joinC.uL,
  ];

  let lineVertex = B;
  if (joinPath.joinIndex >= 2) {
    lineVertex = C;
  }
  let d0 = parents[joinIndex * 2] as v2f;
  let d1 = parents[joinIndex * 2 + 1] as v2f;
  let d = miterPoint(d0, d1);
  // while (depth > 0) {
  //   const isLeftChild = (path & 1) === 0;
  //   path = path >> 1;
  //   depth -= 1;
  //   d0 = select(d, d0, isLeftChild);
  //   d1 = select(d1, d, isLeftChild);
  //   d = bisectNoCheck(d0, d1);
  // }
  return {
    situationIndex: joinB.situationIndex,
    vertexPosition: addMul(lineVertex.position, d, lineVertex.radius),
  };
});
