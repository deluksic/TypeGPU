import { bool, struct, u32, vec2f } from 'typegpu/data';
import type { Infer } from 'typegpu/data';

export type JoinPath = Infer<typeof JoinPath>;
export const JoinPath = struct({
  joinIndex: u32,
  path: u32,
  depth: u32,
});

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

export type CapResult = Infer<typeof CapResult>;
export const CapResult = struct({
  left: vec2f,
  leftForward: vec2f,
  forward: vec2f,
  rightForward: vec2f,
  right: vec2f,
  joinLeft: bool,
  joinRight: bool,
});
