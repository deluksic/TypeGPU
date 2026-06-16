import { struct, u32, vec3f } from 'typegpu/data';

export const ProceduralSphereResult = struct({
  instanceIndex: u32,
  vertex: vec3f,
});
