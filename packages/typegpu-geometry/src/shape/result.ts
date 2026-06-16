import { struct, u32, vec3f } from 'typegpu/data';

export const ProceduralShapeResult = struct({
  instanceIndex: u32,
  vertex: vec3f,
  normal: vec3f,
});

/** @deprecated Use `ProceduralShapeResult`. */
export const ProceduralSphereResult = ProceduralShapeResult;
