import { f32, u32, vec3f, type v3f } from 'typegpu/data';
import { sqrt } from 'typegpu/std';

export function gridBarycentrics(vertexIndex: number, maxSubdivCount: number): v3f {
  'use gpu';
  const level = u32((sqrt(8 * f32(vertexIndex) + 1) - 1) / 2);
  const startIndex = (level * (level + 1)) >> 1;
  const j = f32(vertexIndex - startIndex);
  const i = f32(level) - j;
  const invN = f32(1) / f32(maxSubdivCount);
  return vec3f(f32(1) - (i + j) * invN, i * invN, j * invN);
}
