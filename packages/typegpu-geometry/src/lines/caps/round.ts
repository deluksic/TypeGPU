import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { bisectCcw } from '../../utils.ts';
import { CapResult } from '../types.ts';

export const roundCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  CapResult,
)(
  (right, _dir, left) => {
    const mid = bisectCcw(right, left);
    return CapResult({
      right,
      rightForward: mid,
      forward: mid,
      leftForward: mid,
      left,
      joinRight: true,
      joinLeft: true,
    });
  },
);
