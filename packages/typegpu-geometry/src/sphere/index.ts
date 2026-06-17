export * from './cubesphere.ts';
export * from './icosphere.ts';
export * from './octasphere.ts';
export { spherify } from './spherify.ts';
export * from './segmentSphericalTriangle/index.ts';
export { sphereSlot, sphereObjectIndexSlot } from './slots.ts';

import { cubesphere, cubesphereObjectIndex } from './cubesphere.ts';
import { icosphere, icosphereObjectIndex } from './icosphere.ts';
import { octasphere, octasphereObjectIndex } from './octasphere.ts';

export const proceduralSpheres = { cubesphere, icosphere, octasphere };
export const proceduralSphereObjectIndices = {
  cubesphere: cubesphereObjectIndex,
  icosphere: icosphereObjectIndex,
  octasphere: octasphereObjectIndex,
};
