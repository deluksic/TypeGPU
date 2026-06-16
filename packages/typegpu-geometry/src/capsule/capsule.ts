import tgpu from 'typegpu';
import { arrayOf, f32, u32, type v3f, vec3f } from 'typegpu/data';
import { cos, length, max, min, mix, normalize, select, sin } from 'typegpu/std';
import { uniformArea } from '../sphere/subdivSphericalTriangle/uniformArea.ts';
import { ProceduralShapeResult } from '../shape/result.ts';
import {
  subdivTriangle3,
  subdivTriangleIndexCount,
  subdivTriangleWireframeIndexCount,
} from '../subdividedTriangle.ts';
import { slerpApproxWarp } from '../utils.ts';

const HALF_PI = f32(Math.PI / 2);

/** Four +Y and four −Y octahedron faces. */
export const CAPSULE_CAP_COUNT = 8;
export const CAPSULE_EDGE_COUNT = 4;
export const CAPSULE_EDGE_TRIANGLE_COUNT = CAPSULE_EDGE_COUNT * 2;
export const CAPSULE_EDGE_PATCH_OFFSET = CAPSULE_CAP_COUNT;
export const CAPSULE_PATCH_COUNT = CAPSULE_CAP_COUNT + CAPSULE_EDGE_TRIANGLE_COUNT;

const CAP_OCTANTS = [0, 1, 4, 5, 2, 3, 6, 7] as const;

const capPatchOctants = tgpu.const(arrayOf(u32, CAP_OCTANTS.length), [...CAP_OCTANTS]);

export function capsuleIndexCountPerPatch(subdivisions: number) {
  return subdivTriangleIndexCount(subdivisions);
}

export function capsuleIndexCount(subdivisions: number) {
  return CAPSULE_PATCH_COUNT * capsuleIndexCountPerPatch(subdivisions);
}

export function capsuleWireframeIndexCountPerPatch(subdivisions: number) {
  return subdivTriangleWireframeIndexCount(subdivisions);
}

export function capsuleWireframeIndexCount(subdivisions: number) {
  return CAPSULE_PATCH_COUNT * capsuleWireframeIndexCountPerPatch(subdivisions);
}

export function capsuleInstanceCount(objectCount: number) {
  return objectCount * CAPSULE_PATCH_COUNT;
}

function octantSigns(octant: number): v3f {
  'use gpu';
  return vec3f(
    select(f32(1), f32(-1), (octant & u32(1)) !== u32(0)),
    select(f32(1), f32(-1), (octant & u32(2)) !== u32(0)),
    select(f32(1), f32(-1), (octant & u32(4)) !== u32(0)),
  );
}

function capPatchOctant(capPatchIndex: number): number {
  'use gpu';
  return capPatchOctants.$[capPatchIndex];
}

function edgeSign0(subIndex: number): number {
  'use gpu';
  return select(f32(1), f32(-1), (subIndex & u32(1)) !== u32(0));
}

function edgeSign1(subIndex: number): number {
  'use gpu';
  return select(f32(1), f32(-1), (subIndex & u32(2)) !== u32(0));
}

function edgeCornerArcParam(quadCorner: number): number {
  'use gpu';
  return select(
    f32(0),
    f32(1),
    (quadCorner === u32(1)) || (quadCorner === u32(2)),
  );
}

function projectLinearToCapsuleCylinder(linear: v3f, edgeIndex: number, radius: number): v3f {
  'use gpu';
  const subIndex = edgeIndex & u32(3);
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const t = linear.y;
  const du = linear.x;
  const dv = linear.z;
  const r = f32(radius);
  const radialLen = length(vec3f(du, dv, 0));
  const arcStart = vec3f(s0, 0, 0);
  const arcEnd = vec3f(0, 0, s1);
  const tFromU = f32(1) - du / (s0 * r);
  const tFromV = dv / (s1 * r);
  const arcLinear = min(f32(1), max(f32(0), (tFromU + tFromV) * f32(0.5)));
  const radialUnit = select(
    arcStart,
    normalize(mix(arcStart, arcEnd, slerpApproxWarp(f32(0), arcLinear))),
    radialLen > 0,
  );
  return vec3f(radialUnit.x * r, t, radialUnit.z * r);
}

function capsuleEdgeCorner(
  edgeIndex: number,
  quadCorner: number,
  radius: number,
  cylHalf: number,
): v3f {
  'use gpu';
  const subIndex = edgeIndex & u32(3);
  const s0 = edgeSign0(subIndex);
  const s1 = edgeSign1(subIndex);
  const atTMax = quadCorner >= u32(2);
  const arcParam = edgeCornerArcParam(quadCorner);
  const tVal = select(-f32(cylHalf), f32(cylHalf), atTMax);
  const theta = arcParam * HALF_PI;
  return vec3f(cos(theta) * s0 * radius, tVal, sin(theta) * s1 * radius);
}

function capsuleCapCenter(octant: number, cylHalf: number): v3f {
  'use gpu';
  const signs = octantSigns(octant);
  return vec3f(0, signs.y * f32(cylHalf), 0);
}

function capsuleCap(
  capPatchIndex: number,
  vertexIndex: number,
  subdivCount: number,
  radius: number,
  cylHalf: number,
) {
  'use gpu';
  const octant = capPatchOctant(capPatchIndex);
  const signs = octantSigns(octant);
  const a = vec3f(signs.x, 0, 0);
  const b = vec3f(0, signs.y, 0);
  const c = vec3f(0, 0, signs.z);
  const onSphere = uniformArea(a, b, c, vertexIndex, subdivCount);
  const center = capsuleCapCenter(octant, cylHalf);
  const vertex = center + onSphere * radius;
  return ProceduralShapeResult({
    instanceIndex: 0,
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
  subdivCount: number,
  radius: number,
  cylHalf: number,
) {
  'use gpu';
  const edgeIndex = edgeTriangleIndex >> u32(1);
  const triInEdge = edgeTriangleIndex % u32(2);
  const c0 = capsuleEdgeCorner(edgeIndex, u32(0), radius, cylHalf);
  const c1 = capsuleEdgeCorner(edgeIndex, u32(1), radius, cylHalf);
  const c2 = capsuleEdgeCorner(edgeIndex, u32(2), radius, cylHalf);
  const c3 = capsuleEdgeCorner(edgeIndex, u32(3), radius, cylHalf);
  const a = c0;
  const b = select(c2, c1, triInEdge === u32(0));
  const c = select(c3, c2, triInEdge === u32(0));
  const linear = subdivTriangle3(a, b, c, vertexIndex, subdivCount);
  const vertex = projectLinearToCapsuleCylinder(linear, edgeIndex, radius);
  return ProceduralShapeResult({
    instanceIndex: 0,
    vertex,
    normal: capsuleEdgeNormal(vertex),
  });
}

/**
 * Procedural Y-axis capsule.
 *
 * Eight cap patches are octahedral spherical triangles (`uniformArea`) on the top and bottom
 * hemispheres. Four cylindrical side patches (two triangles each) connect the equators with
 * `slerpApproxWarp` on the radial cross-section.
 */
export function capsule(
  instanceIndex: number,
  vertexIndex: number,
  subdivCount: number,
  radius: number,
  cylHalf: number,
) {
  'use gpu';
  const objectIndex = u32(instanceIndex / CAPSULE_PATCH_COUNT);
  const patchIndex = u32(instanceIndex % CAPSULE_PATCH_COUNT);
  if (patchIndex < CAPSULE_CAP_COUNT) {
    const patch = capsuleCap(patchIndex, vertexIndex, subdivCount, radius, cylHalf);
    return ProceduralShapeResult({
      instanceIndex: objectIndex,
      vertex: patch.vertex,
      normal: patch.normal,
    });
  }
  const patch = capsuleEdge(
    patchIndex - CAPSULE_EDGE_PATCH_OFFSET,
    vertexIndex,
    subdivCount,
    radius,
    cylHalf,
  );
  return ProceduralShapeResult({
    instanceIndex: objectIndex,
    vertex: patch.vertex,
    normal: patch.normal,
  });
}
