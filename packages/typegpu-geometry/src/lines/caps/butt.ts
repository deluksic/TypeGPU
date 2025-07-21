import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { rot90ccw, rot90cw } from '../../utils.ts';
import { CapResult } from '../types.ts';
import { dot, select } from 'typegpu/std';
import { intersectTangent } from '../utils.ts';

export const buttCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  CapResult,
)(
  (right, dir, left) => {
    const shouldJoin = dot(dir, right) < 0;
    const dirRight = rot90cw(dir);
    const dirLeft = rot90ccw(dir);
    const rightForward = select(
      intersectTangent(right, dirRight),
      dirRight,
      shouldJoin,
    );
    const leftForward = select(
      intersectTangent(left, dirLeft),
      dirLeft,
      shouldJoin,
    );
    return CapResult({
      right: select(rightForward, right, shouldJoin),
      rightForward,
      forward: vec2f(0, 0),
      leftForward,
      left: select(leftForward, left, shouldJoin),
    });
  },
);
