import tgpu from 'typegpu';
import { arrayOf, struct, u32, type v3f, vec3f } from 'typegpu/data';
import { dot, normalize, select } from 'typegpu/std';
import { cubeFaceTriangles } from '../cubeFaceTriangles.ts';
import {
  ARC_PATCH_PARAMS_TRI0,
  ARC_PATCH_PARAMS_TRI1,
  arcPatchHeights,
  arcPatchVertex,
} from '../internal/arcPatch.ts';
import { edgeSign0, edgeSign1, octantSigns } from '../internal/octahedronFace.ts';
import { makePatchInstancingHelpers } from '../internal/patchCountHelpers.ts';
import { uniformArea } from '../sphere/segmentSphericalTriangle/uniformArea.ts';
import { ProceduralShapeResult } from '../shape/result.ts';
import { segmentTriangle3 } from '../segmentedTriangle.ts';

export const ROUNDED_BOX_CORNER_COUNT = 8;
export const ROUNDED_BOX_FACE_TRIANGLE_COUNT = 12;
export const ROUNDED_BOX_EDGE_COUNT = 12;
export const ROUNDED_BOX_EDGE_TRIANGLE_COUNT = ROUNDED_BOX_EDGE_COUNT * 2;
export const ROUNDED_BOX_FACE_PATCH_OFFSET = ROUNDED_BOX_CORNER_COUNT;
export const ROUNDED_BOX_EDGE_PATCH_OFFSET =
  ROUNDED_BOX_CORNER_COUNT + ROUNDED_BOX_FACE_TRIANGLE_COUNT;
export const ROUNDED_BOX_PATCH_COUNT =
  ROUNDED_BOX_CORNER_COUNT + ROUNDED_BOX_FACE_TRIANGLE_COUNT + ROUNDED_BOX_EDGE_TRIANGLE_COUNT;

const instancing = makePatchInstancingHelpers(ROUNDED_BOX_PATCH_COUNT);

export const roundedBoxIndexCountPerPatch = instancing.indexCountPerPatch;
export const roundedBoxIndexCount = instancing.indexCount;
export const roundedBoxWireframeIndexCountPerPatch = instancing.wireframeIndexCountPerPatch;
export const roundedBoxWireframeIndexCount = instancing.wireframeIndexCount;
export const roundedBoxInstanceCount = instancing.instanceCount;
export const roundedBoxObjectIndex = instancing.objectIndex;
export const roundedBoxPatchIndex = instancing.patchIndex;

const RoundedBoxEdgeFrame = struct({
  axisDir: vec3f,
  uDir: vec3f,
  vDir: vec3f,
});

function roundedBoxEdgeFrameData(
  axisDir: [number, number, number],
  uDir: [number, number, number],
  vDir: [number, number, number],
) {
  return RoundedBoxEdgeFrame({
    axisDir: vec3f(...axisDir),
    uDir: vec3f(...uDir),
    vDir: vec3f(...vDir),
  });
}

const roundedBoxEdgeFramesData = [
  roundedBoxEdgeFrameData([1, 0, 0], [0, 1, 0], [0, 0, 1]),
  roundedBoxEdgeFrameData([0, 1, 0], [1, 0, 0], [0, 0, 1]),
  roundedBoxEdgeFrameData([0, 0, 1], [1, 0, 0], [0, 1, 0]),
] as const;

const ROUNDED_BOX_EDGE_INDEX_TO_AXIS = [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1] as const;

const roundedBoxEdgePatchFramesData = ROUNDED_BOX_EDGE_INDEX_TO_AXIS.map(
  (axis) => roundedBoxEdgeFramesData[axis],
);

const roundedBoxEdgePatchFrames = tgpu.const(
  arrayOf(RoundedBoxEdgeFrame, roundedBoxEdgePatchFramesData.length),
  roundedBoxEdgePatchFramesData,
);

const roundedBoxFaceNormalsData = [
  vec3f(1, 0, 0),
  vec3f(-1, 0, 0),
  vec3f(0, 1, 0),
  vec3f(0, -1, 0),
  vec3f(0, 0, 1),
  vec3f(0, 0, -1),
];

const roundedBoxFaceNormals = tgpu.const(
  arrayOf(vec3f, roundedBoxFaceNormalsData.length),
  roundedBoxFaceNormalsData,
);

function insetCubeCorner(corner: v3f, halfSize: v3f, cornerRadius: number): v3f {
  'use gpu';
  return vec3f(
    select(-halfSize.x + cornerRadius, halfSize.x - cornerRadius, corner.x > 0),
    select(-halfSize.y + cornerRadius, halfSize.y - cornerRadius, corner.y > 0),
    select(-halfSize.z + cornerRadius, halfSize.z - cornerRadius, corner.z > 0),
  );
}

function roundedBoxEdgeFrame(edgeIndex: number) {
  'use gpu';
  return RoundedBoxEdgeFrame(
    // oxlint-disable-next-line typescript/no-non-null-assertion
    roundedBoxEdgePatchFrames.$[edgeIndex]!
  );
}

function roundedBoxFaceCorner(corner: v3f, halfSize: v3f, cornerRadius: number, faceIndex: number): v3f {
  'use gpu';
  const inset = insetCubeCorner(corner, halfSize, cornerRadius);
  // oxlint-disable-next-line typescript/no-non-null-assertion
  return inset + roundedBoxFaceNormals.$[faceIndex]! * cornerRadius;
}

function roundedBoxEdgeNormal(vertex: v3f, edgeIndex: number, halfSize: v3f, cornerRadius: number): v3f {
  'use gpu';
  const frame = roundedBoxEdgeFrame(edgeIndex);
  const subIndex = edgeIndex & u32(3);
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const uDir = frame.uDir;
  const vDir = frame.vDir;
  const uAxis = s0 * (dot(halfSize, uDir) - cornerRadius);
  const vAxis = s1 * (dot(halfSize, vDir) - cornerRadius);
  const du = dot(vertex, uDir) - uAxis;
  const dv = dot(vertex, vDir) - vAxis;
  return normalize(uDir * du + vDir * dv);
}

function roundedBoxCornerSurface(
  octant: number,
  vertexIndex: number,
  segmentCount: number,
  halfSize: v3f,
  cornerRadius: number,
): v3f {
  'use gpu';
  const signs = octantSigns(octant);
  const a = vec3f(signs.x, 0, 0);
  const b = vec3f(0, signs.y, 0);
  const c = vec3f(0, 0, signs.z);
  const onSphere = uniformArea(a, b, c, vertexIndex, segmentCount);
  const center = signs * (halfSize - vec3f(cornerRadius));
  return center + onSphere * cornerRadius;
}

function roundedBoxCornerNormal(
  octant: number,
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
  segmentCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const vertex = roundedBoxCornerSurface(octant, vertexIndex, segmentCount, halfSize, cornerRadius);
  return ProceduralShapeResult({
    vertex,
    normal: roundedBoxCornerNormal(octant, halfSize, cornerRadius, vertex),
  });
}

function roundedBoxFace(
  faceTriangleIndex: number,
  vertexIndex: number,
  segmentCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const cornerOffset = faceTriangleIndex * 3;
  const faceIndex = faceTriangleIndex >> 1;
  const a = roundedBoxFaceCorner(cubeFaceTriangles.$[cornerOffset] as v3f, halfSize, cornerRadius, faceIndex);
  const b = roundedBoxFaceCorner(
    cubeFaceTriangles.$[cornerOffset + 1] as v3f,
    halfSize,
    cornerRadius,
    faceIndex,
  );
  const c = roundedBoxFaceCorner(
    cubeFaceTriangles.$[cornerOffset + 2] as v3f,
    halfSize,
    cornerRadius,
    faceIndex,
  );
  const normal = roundedBoxFaceNormals.$[faceIndex] as v3f
  return ProceduralShapeResult({
    vertex: segmentTriangle3(a, b, c, vertexIndex, segmentCount),
    normal,
  });
}

function roundedBoxEdge(
  edgeTriangleIndex: number,
  vertexIndex: number,
  segmentCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const edgeIndex = edgeTriangleIndex >> 1;
  const triInEdge = edgeTriangleIndex % 2;
  const frame = roundedBoxEdgeFrame(edgeIndex);
  const subIndex = edgeIndex & 0b11;
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const uDir = frame.uDir;
  const vDir = frame.vDir;
  const axisDir = frame.axisDir;
  const ou = dot(halfSize, uDir);
  const ov = dot(halfSize, vDir);
  const center = uDir * (s0 * (ou - cornerRadius)) + vDir * (s1 * (ov - cornerRadius));
  const ot = dot(halfSize, axisDir);
  const hMin = -ot + cornerRadius;
  const hMax = ot - cornerRadius;
  const arcParams = select(ARC_PATCH_PARAMS_TRI1, ARC_PATCH_PARAMS_TRI0, triInEdge === u32(0));
  const heights = arcPatchHeights(triInEdge, hMin, hMax);
  const vertex = arcPatchVertex(
    vertexIndex,
    segmentCount,
    center,
    axisDir,
    uDir * s0,
    vDir * s1,
    arcParams,
    heights,
    cornerRadius,
  );
  return ProceduralShapeResult({
    vertex,
    normal: roundedBoxEdgeNormal(vertex, edgeIndex, halfSize, cornerRadius),
  });
}

/**
 * Procedural rounded box.
 *
 * Eight corner patches use an octahedral spherical triangle (`uniformArea`).
 * Edge patches use swept-arc developable patches (`arcPatchVertex`).
 * Six face pairs use flat segmented triangles on the outer faces, inset by cornerRadius
 * on the tangent axes.
 * Shared `segmentTriangleIndices(maxSegmentCount)` index buffer; draw a prefix per patch.
 */
export function roundedBox(
  instanceIndex: number,
  vertexIndex: number,
  segmentCount: number,
  halfSize: v3f,
  cornerRadius: number,
) {
  'use gpu';
  const patchIndex = roundedBoxPatchIndex(instanceIndex);
  if (patchIndex < ROUNDED_BOX_CORNER_COUNT) {
    return roundedBoxCorner(patchIndex, vertexIndex, segmentCount, halfSize, cornerRadius);
  }
  if (patchIndex < ROUNDED_BOX_EDGE_PATCH_OFFSET) {
    return roundedBoxFace(
      patchIndex - ROUNDED_BOX_FACE_PATCH_OFFSET,
      vertexIndex,
      segmentCount,
      halfSize,
      cornerRadius,
    );
  }
  return roundedBoxEdge(
    patchIndex - ROUNDED_BOX_EDGE_PATCH_OFFSET,
    vertexIndex,
    segmentCount,
    halfSize,
    cornerRadius,
  );
}
