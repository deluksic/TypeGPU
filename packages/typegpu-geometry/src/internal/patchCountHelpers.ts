import { u32 } from 'typegpu/data';
import {
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndexCount,
} from '../segmentedTriangle.ts';

export function makePatchInstancingHelpers(patchCount: number) {
  function objectIndex(drawInstanceIndex: number) {
    'use gpu';
    return u32(drawInstanceIndex / patchCount);
  }

  function patchIndex(drawInstanceIndex: number) {
    'use gpu';
    return u32(drawInstanceIndex % patchCount);
  }

  return {
    indexCountPerPatch: (segmentCount: number) => segmentTriangleIndexCount(segmentCount),
    indexCount: (segmentCount: number) => patchCount * segmentTriangleIndexCount(segmentCount),
    wireframeIndexCountPerPatch: (segmentCount: number) =>
      segmentTriangleWireframeIndexCount(segmentCount),
    wireframeIndexCount: (segmentCount: number) =>
      patchCount * segmentTriangleWireframeIndexCount(segmentCount),
    instanceCount: (objectCount: number) => patchCount * objectCount,
    objectIndex,
    patchIndex,
  };
}
