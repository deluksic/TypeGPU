export * from './cubesphere.ts';
export * from './icosphere.ts';
export * from './octasphere.ts';
export { spherify } from './spherify.ts';
export * from './subdivSphericalTriangle/index.ts';
export { ProceduralShapeResult, ProceduralSphereResult } from './result.ts';
export { sphereSlot } from './slots.ts';

import { cubesphere } from './cubesphere.ts';
import { icosphere } from './icosphere.ts';
import { octasphere } from './octasphere.ts';

export const proceduralSpheres = { cubesphere, icosphere, octasphere };
