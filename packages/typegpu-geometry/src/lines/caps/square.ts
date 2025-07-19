import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { JoinResult } from '../types.ts';
import { miterPointNoCheck } from '../utils.ts';

export const squareCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  JoinResult,
)(
  (dir, a, b) => {
    const miterA = miterPointNoCheck(a, dir);
    const miterB = miterPointNoCheck(dir, b);
    return JoinResult({
      uL: miterB,
      u: dir,
      uR: miterA,
      c: dir,
      dL: miterA,
      d: dir,
      dR: miterB,
      joinUL: false,
      joinUR: false,
      joinDL: false,
      joinDR: false,
      situationIndex: 0,
    });
  },
);
