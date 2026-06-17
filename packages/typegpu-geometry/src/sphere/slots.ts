import tgpu from 'typegpu';
import { icosphere, icosphereObjectIndex } from './icosphere.ts';

export const sphereSlot = tgpu.slot(icosphere);
export const sphereObjectIndexSlot = tgpu.slot(icosphereObjectIndex);
