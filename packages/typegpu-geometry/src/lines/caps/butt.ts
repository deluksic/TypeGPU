import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { rot90ccw } from '../../utils.ts';
import { JoinResult } from '../types.ts';
import { dot, mul } from 'typegpu/std';

const project = tgpu.fn([vec2f, vec2f], vec2f)((a, n) => {
  const cos_ = dot(a, n);
  return mul(n, 1 / cos_);
});

export const buttCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  JoinResult,
)(
  (dir, a, b) => {
    const zeroAxis = rot90ccw(dir);
    const miterA = project(a, zeroAxis);
    const miterB = project(b, zeroAxis);
    return JoinResult({
      uL: miterB,
      u: vec2f(0, 0),
      uR: miterA,
      c: vec2f(0, 0),
      dL: miterA,
      d: vec2f(0, 0),
      dR: miterB,
      joinUL: false,
      joinUR: false,
      joinDL: false,
      joinDR: false,
      situationIndex: 0,
    });
  },
);
