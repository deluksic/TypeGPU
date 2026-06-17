import tgpu from 'typegpu';
import { arrayOf, f32, u32, type v3f, vec3f } from 'typegpu/data';
import { normalize, select } from 'typegpu/std';
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

/** Four +Y and four −Y octahedron faces. */
export const CAPSULE_CAP_COUNT = 8;
export const CAPSULE_EDGE_COUNT = 4;
export const CAPSULE_EDGE_TRIANGLE_COUNT = CAPSULE_EDGE_COUNT * 2;
export const CAPSULE_EDGE_PATCH_OFFSET = CAPSULE_CAP_COUNT;
export const CAPSULE_PATCH_COUNT = CAPSULE_CAP_COUNT + CAPSULE_EDGE_TRIANGLE_COUNT;

const CAP_OCTANTS = [0, 1, 4, 5, 2, 3, 6, 7] as const;

const capPatchOctants = tgpu.const(arrayOf(u32, CAP_OCTANTS.length), [...CAP_OCTANTS]);

const instancing = makePatchInstancingHelpers(CAPSULE_PATCH_COUNT);

export const capsuleIndexCountPerPatch = instancing.indexCountPerPatch;
export const capsuleIndexCount = instancing.indexCount;
export const capsuleWireframeIndexCountPerPatch = instancing.wireframeIndexCountPerPatch;
export const capsuleWireframeIndexCount = instancing.wireframeIndexCount;
export const capsuleInstanceCount = instancing.instanceCount;
export const capsuleObjectIndex = instancing.objectIndex;
export const capsulePatchIndex = instancing.patchIndex;

const CAPSULE_AXIS_DIR = vec3f(0, 1, 0);

function capPatchOctant(capPatchIndex: number): number {
  'use gpu';
  // oxlint-disable-next-line typescript/no-non-null-assertion
  return capPatchOctants.$[capPatchIndex]!;
}

function capsuleCapCenter(octant: number, cylHalf: number): v3f {
  'use gpu';
  const signs = octantSigns(octant);
  return vec3f(0, signs.y * f32(cylHalf), 0);
}

function capsuleCap(
  capPatchIndex: number,
  vertexIndex: number,
  segmentCount: number,
  radius: number,
  cylHalf: number,
) {
  'use gpu';
  const octant = capPatchOctant(capPatchIndex);
  const signs = octantSigns(octant);
  const a = vec3f(signs.x, 0, 0);
  const b = vec3f(0, signs.y, 0);
  const c = vec3f(0, 0, signs.z);
  const onSphere = uniformArea(a, b, c, vertexIndex, segmentCount);
  const center = capsuleCapCenter(octant, cylHalf);
  const vertex = center + onSphere * radius;
  return ProceduralShapeResult({
    vertex,
    normal: normalize(vertex - center),
  });
}

function capsuleEdgeNormal(vertex: v3f): v3f {
  'use gpu';
  return normalize(vec3f(vertex.x, 0, vertex.z));
}

function capsuleEdge(
  edgeTriangleIndex: number,
  vertexIndex: number,
  segmentCount: number,
  radius: number,
  cylHalf: number,
) {
  'use gpu';
  const edgeIndex = edgeTriangleIndex >> 1;
  const triInEdge = edgeTriangleIndex % 2;
  const subIndex = edgeIndex & 0b11;
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const hMin = -f32(cylHalf);
  const hMax = f32(cylHalf);
  const arcParams = select(ARC_PATCH_PARAMS_TRI1, ARC_PATCH_PARAMS_TRI0, triInEdge === 0);
  const heights = arcPatchHeights(triInEdge, hMin, hMax);
  const vertex = arcPatchVertex(
    vertexIndex,
    segmentCount,
    vec3f(0, 0, 0),
    CAPSULE_AXIS_DIR,
    vec3f(s0, 0, 0),
    vec3f(0, 0, s1),
    arcParams,
    heights,
    radius,
  );
  return ProceduralShapeResult({
    vertex,
    normal: capsuleEdgeNormal(vertex),
  });
}

/**
 * Procedural Y-axis capsule.
 *
 * Eight cap patches are octahedral spherical triangles (`uniformArea`) on the top and bottom
 * hemispheres. Four cylindrical side patches (two triangles each) connect the equators via
 * swept-arc developable patches (`arcPatchVertex`).
 */
export function capsule(
  instanceIndex: number,
  vertexIndex: number,
  segmentCount: number,
  radius: number,
  cylHalf: number,
) {
  'use gpu';
  const patchIndex = capsulePatchIndex(instanceIndex);
  if (patchIndex < CAPSULE_CAP_COUNT) {
    return capsuleCap(patchIndex, vertexIndex, segmentCount, radius, cylHalf);
  }
  return capsuleEdge(
    patchIndex - CAPSULE_EDGE_PATCH_OFFSET,
    vertexIndex,
    segmentCount,
    radius,
    cylHalf,
  );
}
