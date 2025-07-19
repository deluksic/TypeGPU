import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';
import { bisectCcw } from '../../utils.ts';
import { JoinResult } from '../types.ts';

export const roundCap = tgpu.fn(
  [vec2f, vec2f, vec2f],
  JoinResult,
)(
  (_dir, a, b) => {
    const mid = bisectCcw(a, b);
    return JoinResult({
      uL: b,
      u: mid,
      uR: a,
      c: mid,
      dL: a,
      d: mid,
      dR: b,
      joinUL: true,
      joinUR: true,
      joinDL: true,
      joinDR: true,
      situationIndex: 0,
    });
  },
);
