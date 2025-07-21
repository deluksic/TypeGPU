import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { CapResult } from '../types.ts';

export const roundCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  CapResult,
)(
  (right, dir, left) => {
    return CapResult({
      right,
      rightForward: dir,
      forward: dir,
      leftForward: dir,
      left,
    });
  },
);
