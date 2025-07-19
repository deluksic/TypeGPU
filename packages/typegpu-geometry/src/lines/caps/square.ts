import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { CapResult } from '../types.ts';
import { miterPointNoCheck } from '../utils.ts';
import { add, dot, select } from 'typegpu/std';
import { rot90ccw, rot90cw } from '../../utils.ts';

export const squareCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  CapResult,
)(
  (right, dir, left) => {
    const shouldJoin = dot(dir, right) < 0;
    const dirRight = rot90cw(dir);
    const dirLeft = rot90ccw(dir);
    const miterR = miterPointNoCheck(right, dirRight);
    const miterL = miterPointNoCheck(dirLeft, left);
    const rightForward = select(
      miterPointNoCheck(right, dir),
      add(dir, dirRight),
      shouldJoin,
    );
    const leftForward = select(
      miterPointNoCheck(dir, left),
      add(dir, dirLeft),
      shouldJoin,
    );
    return CapResult({
      right: miterR,
      rightForward,
      forward: dir,
      leftForward,
      left: miterL,
      joinRight: true,
      joinLeft: true,
    });
  },
);
