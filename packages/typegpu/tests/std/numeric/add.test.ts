import { describe, expect, it } from 'vitest';
import {
  vec2f,
  vec2i,
  vec2u,
  vec3f,
  vec3i,
  vec3u,
  vec4f,
  vec4i,
  vec4u,
} from '../../../src/data/index.ts';
import { add } from '../../../src/std/index.ts';

describe('add', () => {
  it('computes sum of two vec2f', () => {
    expect(add(vec2f(0, 0), vec2f(0, 0))).toStrictEqual(vec2f(0, 0));
    expect(add(vec2f(1.2, 0.1), vec2f(1.0, 0.5))).toStrictEqual(
      vec2f(2.2, 0.6),
    );
    expect(add(vec2f(-1.5, 1), vec2f(1, 0.1))).toStrictEqual(vec2f(-0.5, 1.1));
  });

  it('computes sum of two vec2u', () => {
    expect(add(vec2u(0, 0), vec2u(0, 0))).toStrictEqual(vec2u(0, 0));
    expect(add(vec2u(1, 0), vec2u(1, 2))).toStrictEqual(vec2u(2, 2));
  });

  it('computes sum of two vec2i', () => {
    expect(add(vec2i(0, 0), vec2i(0, 0))).toStrictEqual(vec2i(0, 0));
    expect(add(vec2i(1, 0), vec2i(1, 0))).toStrictEqual(vec2i(2, 0));
    expect(add(vec2i(-1, 1), vec2i(1, 0))).toStrictEqual(vec2i(0, 1));
  });

  it('computes sum of two vec3f', () => {
    expect(add(vec3f(1.5, 2, 3), vec3f(-1.5, -2, -3))).toStrictEqual(
      vec3f(0, 0, 0),
    );
    expect(add(vec3f(1, 1, 1), vec3f(2, 3, 4))).toStrictEqual(vec3f(3, 4, 5));
    expect(add(vec3f(1.5), vec3f(2))).toStrictEqual(vec3f(3.5));
  });

  it('computes sum of two vec3u', () => {
    expect(add(vec3u(1, 1, 1), vec3u(2, 3, 4))).toStrictEqual(vec3u(3, 4, 5));
    expect(add(vec3u(1), vec3u(2))).toStrictEqual(vec3u(3));
  });

  it('computes sum of two vec3i', () => {
    expect(add(vec3i(1, 2, 3), vec3i(-1, -2, -3))).toStrictEqual(
      vec3i(0, 0, 0),
    );
    expect(add(vec3i(1, 1, 1), vec3i(2, 3, 4))).toStrictEqual(vec3i(3, 4, 5));
    expect(add(vec3i(1), vec3i(2))).toStrictEqual(vec3i(3));
  });

  it('computes sum of two vec4f', () => {
    expect(add(vec4f(1.5, 2, 3, 4), vec4f(-1.5, -2, -3, -4))).toStrictEqual(
      vec4f(0, 0, 0, 0),
    );
    expect(add(vec4f(1, 1, 1, 1), vec4f(2, 3.5, 4, 5))).toStrictEqual(
      vec4f(3, 4.5, 5, 6),
    );
    expect(add(vec4f(1), vec4f(2.5))).toStrictEqual(vec4f(3.5));
  });

  it('computes sum of two vec4u', () => {
    expect(add(vec4u(1, 1, 1, 1), vec4u(2, 3, 4, 5))).toStrictEqual(
      vec4u(3, 4, 5, 6),
    );
    expect(add(vec4u(1), vec4u(2))).toStrictEqual(vec4u(3));
  });

  it('computes sum of two vec4i', () => {
    expect(add(vec4i(1, 2, 3, 4), vec4i(-1, -2, -3, -4))).toStrictEqual(
      vec4i(0, 0, 0, 0),
    );
    expect(add(vec4i(1, 1, 1, 1), vec4i(2, 3, 4, 5))).toStrictEqual(
      vec4i(3, 4, 5, 6),
    );
    expect(add(vec4i(1), vec4i(2))).toStrictEqual(vec4i(3));
  });
});
