import { f32, u32, type v3f, vec3f } from 'typegpu/data';
import { cos, length, max, min, mix, normalize, select, sin } from 'typegpu/std';
import { cubeFaceTriangles } from '../cubeFaceTriangles.ts';
import { uniformArea } from '../sphere/subdivSphericalTriangle/uniformArea.ts';
import { ProceduralShapeResult } from '../shape/result.ts';
import {
  subdivTriangle3,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../subdividedTriangle.ts';
import { slerpApproxWarp } from '../utils.ts';

const HALF_PI = f32(Math.PI / 2);

export const ROUNDED_BOX_CORNER_COUNT = 8;
export const ROUNDED_BOX_FACE_TRIANGLE_COUNT = 12;
export const ROUNDED_BOX_EDGE_COUNT = 12;
export const ROUNDED_BOX_EDGE_TRIANGLE_COUNT = ROUNDED_BOX_EDGE_COUNT * 2;
export const ROUNDED_BOX_FACE_PATCH_OFFSET = ROUNDED_BOX_CORNER_COUNT;
export const ROUNDED_BOX_EDGE_PATCH_OFFSET =
  ROUNDED_BOX_CORNER_COUNT + ROUNDED_BOX_FACE_TRIANGLE_COUNT;
export const ROUNDED_BOX_PATCH_COUNT =
  ROUNDED_BOX_CORNER_COUNT + ROUNDED_BOX_FACE_TRIANGLE_COUNT + ROUNDED_BOX_EDGE_TRIANGLE_COUNT;

export function roundedBoxIndexCountPerPatch(subdivisions: number) {
  return subdivTriangleIndexCount(subdivisions);
}

export function roundedBoxIndexCount(subdivisions: number) {
  return ROUNDED_BOX_PATCH_COUNT * roundedBoxIndexCountPerPatch(subdivisions);
}

export function roundedBoxWireframeIndexCountPerPatch(subdivisions: number) {
  return subdivTriangleWireframeIndexCount(subdivisions);
}

export function roundedBoxWireframeIndexCount(subdivisions: number) {
  return ROUNDED_BOX_PATCH_COUNT * roundedBoxWireframeIndexCountPerPatch(subdivisions);
}

export function roundedBoxInstanceCount(objectCount: number) {
  return objectCount * ROUNDED_BOX_PATCH_COUNT;
}

function octantSigns(octant: number): v3f {
  'use gpu';
  return vec3f(
    select(f32(1), f32(-1), (octant & u32(1)) !== u32(0)),
    select(f32(1), f32(-1), (octant & u32(2)) !== u32(0)),
    select(f32(1), f32(-1), (octant & u32(4)) !== u32(0)),
  );
}

function insetCubeCorner(corner: v3f, halfSize: v3f, cornerRadius: number): v3f {
  'use gpu';
  return vec3f(
    select(-halfSize.x + cornerRadius, halfSize.x - cornerRadius, corner.x > 0),
    select(-halfSize.y + cornerRadius, halfSize.y - cornerRadius, corner.y > 0),
    select(-halfSize.z + cornerRadius, halfSize.z - cornerRadius, corner.z > 0),
  );
}

function edgeIndexToAxis(edgeIndex: number): number {
  'use gpu';
  if (edgeIndex < u32(4)) {
    return u32(2);
  }
  if (edgeIndex < u32(8)) {
    return u32(0);
  }
  return u32(1);
}

function edgeSign0(subIndex: number): number {
  'use gpu';
  return select(f32(1), f32(-1), (subIndex & u32(1)) !== u32(0));
}

function edgeSign1(subIndex: number): number {
  'use gpu';
  return select(f32(1), f32(-1), (subIndex & u32(2)) !== u32(0));
}

function edgeOuterT(axis: number, halfSize: v3f): number {
  'use gpu';
  if (axis === u32(0)) {
    return halfSize.x;
  }
  if (axis === u32(1)) {
    return halfSize.y;
  }
  return halfSize.z;
}

function edgeOuterU(axis: number, halfSize: v3f): number {
  'use gpu';
  if (axis === u32(0)) {
    return halfSize.y;
  }
  if (axis === u32(1)) {
    return halfSize.x;
  }
  return halfSize.x;
}

function edgeOuterV(axis: number, halfSize: v3f): number {
  'use gpu';
  if (axis === u32(0)) {
    return halfSize.z;
  }
  if (axis === u32(1)) {
    return halfSize.z;
  }
  return halfSize.y;
}

function composeAxis(axis: number, t: number, u: number, v: number): v3f {
  'use gpu';
  if (axis === u32(0)) {
    return vec3f(t, u, v);
  }
  if (axis === u32(1)) {
    return vec3f(u, t, v);
  }
  return vec3f(u, v, t);
}

function axisT(axis: number, point: v3f): number {
  'use gpu';
  if (axis === u32(0)) {
    return point.x;
  }
  if (axis === u32(1)) {
    return point.y;
  }
  return point.z;
}

function axisU(axis: number, point: v3f): number {
  'use gpu';
  if (axis === u32(0)) {
    return point.y;
  }
  if (axis === u32(1)) {
    return point.x;
  }
  return point.x;
}

function axisV(axis: number, point: v3f): number {
  'use gpu';
  if (axis === u32(0)) {
    return point.z;
  }
  if (axis === u32(1)) {
    return point.z;
  }
  return point.y;
}

function edgeRadialVector(axis: number, du: number, dv: number): v3f {
  'use gpu';
  if (axis === u32(0)) {
    return vec3f(0, du, dv);
  }
  if (axis === u32(1)) {
    return vec3f(du, 0, dv);
  }
  return vec3f(du, dv, 0);
}

function edgeCornerArcParam(quadCorner: number): number {
  'use gpu';
  return select(
    f32(0),
    f32(1),
    (quadCorner === u32(1)) || (quadCorner === u32(2)),
  );
}

/** Linear 3D point projected onto the edge cylinder via `slerpApproxWarp` in the cross-section. */
function projectLinearToEdgeCylinder(
  linear: v3f,
  edgeIndex: number,
  halfSize: v3f,
  cornerRadius: number,
): v3f {
  'use gpu';
  const axis = edgeIndexToAxis(edgeIndex);
  const subIndex = edgeIndex & u32(3);
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const ou = edgeOuterU(axis, halfSize);
  const ov = edgeOuterV(axis, halfSize);
  const uAxis = s0 * (ou - cornerRadius);
  const vAxis = s1 * (ov - cornerRadius);
  const t = axisT(axis, linear);
  const du = axisU(axis, linear) - uAxis;
  const dv = axisV(axis, linear) - vAxis;
  const r = f32(cornerRadius);
  const radialLen = length(edgeRadialVector(axis, du, dv));
  const arcStart = normalize(edgeRadialVector(axis, s0, 0));
  const arcEnd = normalize(edgeRadialVector(axis, 0, s1));
  const tFromU = f32(1) - du / (s0 * r);
  const tFromV = dv / (s1 * r);
  const arcLinear = min(f32(1), max(f32(0), (tFromU + tFromV) * f32(0.5)));
  const radialUnit = select(
    arcStart,
    normalize(mix(arcStart, arcEnd, slerpApproxWarp(f32(0), arcLinear))),
    radialLen > 0,
  );
  const u = uAxis + axisU(axis, radialUnit) * r;
  const v = vAxis + axisV(axis, radialUnit) * r;
  return composeAxis(axis, t, u, v);
}

function edgeCylinderPoint(
  edgeIndex: number,
  tVal: number,
  arcParam: number,
  halfSize: v3f,
  cornerRadius: number,
): v3f {
  'use gpu';
  const axis = edgeIndexToAxis(edgeIndex);
  const subIndex = edgeIndex & u32(3);
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const ou = edgeOuterU(axis, halfSize);
  const ov = edgeOuterV(axis, halfSize);
  const uAxis = s0 * (ou - cornerRadius);
  const vAxis = s1 * (ov - cornerRadius);
  const theta = arcParam * HALF_PI;
  const u = uAxis + cos(theta) * s0 * cornerRadius;
  const v = vAxis + sin(theta) * s1 * cornerRadius;
  return composeAxis(axis, tVal, u, v);
}

function edgeQuarterCirclePoint(
  edgeIndex: number,
  atTMax: boolean,
  arcParam: number,
  halfSize: v3f,
  cornerRadius: number,
): v3f {
  'use gpu';
  const axis = edgeIndexToAxis(edgeIndex);
  const ot = edgeOuterT(axis, halfSize);
  const tMin = -ot + cornerRadius;
  const tMax = ot - cornerRadius;
  const tVal = select(tMin, tMax, atTMax);
  return edgeCylinderPoint(edgeIndex, tVal, arcParam, halfSize, cornerRadius);
}

function roundedBoxFaceNormal(faceIndex: number): v3f {
  'use gpu';
  if (faceIndex === u32(0)) {
    return vec3f(1, 0, 0);
  }
  if (faceIndex === u32(1)) {
    return vec3f(-1, 0, 0);
  }
  if (faceIndex === u32(2)) {
    return vec3f(0, 1, 0);
  }
  if (faceIndex === u32(3)) {
    return vec3f(0, -1, 0);
  }
  if (faceIndex === u32(4)) {
    return vec3f(0, 0, 1);
  }
  return vec3f(0, 0, -1);
}

function roundedBoxFaceCorner(corner: v3f, halfSize: v3f, cornerRadius: number, faceIndex: number): v3f {
  'use gpu';
  const inset = insetCubeCorner(corner, halfSize, cornerRadius);
  return inset + roundedBoxFaceNormal(faceIndex) * cornerRadius;
}

function roundedBoxEdgeNormal(vertex: v3f, edgeIndex: number, halfSize: v3f, cornerRadius: number): v3f {
  'use gpu';
  const axis = edgeIndexToAxis(edgeIndex);
  const subIndex = edgeIndex & u32(3);
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const ou = edgeOuterU(axis, halfSize);
  const ov = edgeOuterV(axis, halfSize);
  const uAxis = s0 * (ou - cornerRadius);
  const vAxis = s1 * (ov - cornerRadius);
  const du = axisU(axis, vertex) - uAxis;
  const dv = axisV(axis, vertex) - vAxis;
  return normalize(edgeRadialVector(axis, du, dv));
}

function roundedBoxCornerSurface(
  octant: number,
  vertexIndex: number,
  subdivCount: number,
  halfSize: v3f,
  cornerRadius: number,
): v3f {
  'use gpu';
  const signs = octantSigns(octant);
  const a = vec3f(signs.x, 0, 0);
  const b = vec3f(0, signs.y, 0);
  const c = vec3f(0, 0, signs.z);
  const onSphere = uniformArea(a, b, c, vertexIndex, subdivCount);
  const center = signs * (halfSize - vec3f(cornerRadius));
  return center + onSphere * cornerRadius;
}

function roundedBoxCornerNormal(
  octant: number,
  vertexIndex: number,
  subdivCount: number,
  halfSize: v3f,
  cornerRadius: number,
  vertex: v3f,
): v3f {
  'use gpu';
  const signs = octantSigns(octant);
  const center = signs * (halfSize - vec3f(cornerRadius));
  return normalize(vertex - center);
}

function roundedBoxCorner(
  octant: number,
  vertexIndex: number,
  subdivCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const vertex = roundedBoxCornerSurface(octant, vertexIndex, subdivCount, halfSize, cornerRadius);
  return ProceduralShapeResult({
    instanceIndex: 0,
    vertex,
    normal: roundedBoxCornerNormal(
      octant,
      vertexIndex,
      subdivCount,
      halfSize,
      cornerRadius,
      vertex,
    ),
  });
}

function roundedBoxFace(
  faceTriangleIndex: number,
  vertexIndex: number,
  subdivCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const cornerOffset = faceTriangleIndex * u32(3);
  const faceIndex = faceTriangleIndex >> u32(1);
  const a = roundedBoxFaceCorner(cubeFaceTriangles.$[cornerOffset] as v3f, halfSize, cornerRadius, faceIndex);
  const b = roundedBoxFaceCorner(
    cubeFaceTriangles.$[cornerOffset + u32(1)] as v3f,
    halfSize,
    cornerRadius,
    faceIndex,
  );
  const c = roundedBoxFaceCorner(
    cubeFaceTriangles.$[cornerOffset + u32(2)] as v3f,
    halfSize,
    cornerRadius,
    faceIndex,
  );
  return ProceduralShapeResult({
    instanceIndex: 0,
    vertex: subdivTriangle3(a, b, c, vertexIndex, subdivCount),
    normal: roundedBoxFaceNormal(faceIndex),
  });
}

function roundedBoxEdgeCorner(
  edgeIndex: number,
  quadCorner: number,
  halfSize: v3f,
  cornerRadius: number,
): v3f {
  'use gpu';
  const atTMax = quadCorner >= u32(2);
  return edgeQuarterCirclePoint(
    edgeIndex,
    atTMax,
    edgeCornerArcParam(quadCorner),
    halfSize,
    cornerRadius,
  );
}

function roundedBoxEdge(
  edgeTriangleIndex: number,
  vertexIndex: number,
  subdivCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const edgeIndex = edgeTriangleIndex >> u32(1);
  const triInEdge = edgeTriangleIndex % u32(2);
  const c0 = roundedBoxEdgeCorner(edgeIndex, u32(0), halfSize, cornerRadius);
  const c1 = roundedBoxEdgeCorner(edgeIndex, u32(1), halfSize, cornerRadius);
  const c2 = roundedBoxEdgeCorner(edgeIndex, u32(2), halfSize, cornerRadius);
  const c3 = roundedBoxEdgeCorner(edgeIndex, u32(3), halfSize, cornerRadius);
  const a = c0;
  const b = select(c2, c1, triInEdge === u32(0));
  const c = select(c3, c2, triInEdge === u32(0));
  const linear = subdivTriangle3(a, b, c, vertexIndex, subdivCount);
  const vertex = projectLinearToEdgeCylinder(linear, edgeIndex, halfSize, cornerRadius);
  return ProceduralShapeResult({
    instanceIndex: 0,
    vertex,
    normal: roundedBoxEdgeNormal(vertex, edgeIndex, halfSize, cornerRadius),
  });
}

/**
 * Procedural rounded box.
 *
 * Eight corner patches use an octahedral spherical triangle (`uniformArea`).
 * Edge patches use linear barycentrics, then `slerpApproxWarp` on the quarter-arc cross-section
 * (`mix` + normalize, not chord normalize).
 * Six face pairs use flat subdivided triangles on the outer faces, inset by cornerRadius
 * on the tangent axes.
 * Shared `subdivTriangleIndices(maxSubdiv)` index buffer; draw a prefix per patch.
 */
export function roundedBox(
  instanceIndex: number,
  vertexIndex: number,
  subdivCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const objectIndex = u32(instanceIndex / ROUNDED_BOX_PATCH_COUNT);
  const patchIndex = u32(instanceIndex % ROUNDED_BOX_PATCH_COUNT);
  if (patchIndex < ROUNDED_BOX_CORNER_COUNT) {
    const patch = roundedBoxCorner(patchIndex, vertexIndex, subdivCount, halfSize, cornerRadius);
    return ProceduralShapeResult({
      instanceIndex: objectIndex,
      vertex: patch.vertex,
      normal: patch.normal,
    });
  }
  if (patchIndex < ROUNDED_BOX_EDGE_PATCH_OFFSET) {
    const patch = roundedBoxFace(
      patchIndex - ROUNDED_BOX_FACE_PATCH_OFFSET,
      vertexIndex,
      subdivCount,
      halfSize,
      cornerRadius,
    );
    return ProceduralShapeResult({
      instanceIndex: objectIndex,
      vertex: patch.vertex,
      normal: patch.normal,
    });
  }
  const patch = roundedBoxEdge(
    patchIndex - ROUNDED_BOX_EDGE_PATCH_OFFSET,
    vertexIndex,
    subdivCount,
    halfSize,
    cornerRadius,
  );
  return ProceduralShapeResult({
    instanceIndex: objectIndex,
    vertex: patch.vertex,
    normal: patch.normal,
  });
}
