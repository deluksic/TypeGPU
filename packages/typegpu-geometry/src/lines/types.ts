import { bool, struct, u32, vec2f } from 'typegpu/data';
import type { Infer } from 'typegpu/data';

export type JoinResult = Infer<typeof JoinResult>;
export const JoinResult = struct({
  uL: vec2f,
  u: vec2f,
  uR: vec2f,
  c: vec2f,
  dL: vec2f,
  d: vec2f,
  dR: vec2f,
  situationIndex: u32,
  joinUL: bool,
  joinDL: bool,
  joinUR: bool,
  joinDR: bool,
});
