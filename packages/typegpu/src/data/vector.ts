import { inGPUMode } from '../gpuMode';
import {
  Vec2fImpl,
  Vec2hImpl,
  Vec2iImpl,
  Vec2uImpl,
  Vec3fImpl,
  Vec3hImpl,
  Vec3iImpl,
  Vec3uImpl,
  Vec4fImpl,
  Vec4hImpl,
  Vec4iImpl,
  Vec4uImpl,
  type VecBase,
} from './vectorImpl';
import type {
  Vec2f,
  Vec2h,
  Vec2i,
  Vec2u,
  Vec3f,
  Vec3h,
  Vec3i,
  Vec3u,
  Vec4f,
  Vec4h,
  Vec4i,
  Vec4u,
  v2f,
  v2h,
  v2i,
  v2u,
  v3f,
  v3h,
  v3i,
  v3u,
  v4f,
  v4h,
  v4i,
  v4u,
} from './wgslTypes';

// --------------
// Implementation
// --------------

type VecSchemaBase<TValue> = {
  readonly type: string;
  readonly '~repr': TValue;
};

function makeVecSchema<TValue>(
  VecImpl: new (...args: number[]) => VecBase,
): VecSchemaBase<TValue> & ((...args: number[]) => TValue) {
  const { kind: type, length: componentCount } = new VecImpl();

  const construct = (...args: number[]): TValue => {
    const values = args; // TODO: Allow users to pass in vectors that fill part of the values.

    if (inGPUMode()) {
      return `${type}(${values.join(', ')})` as unknown as TValue;
    }

    if (values.length <= 1 || values.length === componentCount) {
      return new VecImpl(...values) as TValue;
    }

    throw new Error(
      `'${type}' constructor called with invalid number of arguments.`,
    );
  };

  return Object.assign(construct, { type, '~repr': undefined as TValue });
}

// ----------
// Public API
// ----------

/**
 * Type of the `d.vec2f` object/function: vector data type schema/constructor
 */
export type NativeVec2f = Vec2f & { '~exotic': Vec2f } & ((
    x: number,
    y: number,
  ) => v2f) &
  ((xy: number) => v2f) &
  (() => v2f);

/**
 *
 * Schema representing vec2f - a vector with 2 elements of type f32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec2f(); // (0.0, 0.0)
 * const vector = d.vec2f(1); // (1.0, 1.0)
 * const vector = d.vec2f(0.5, 0.1); // (0.5, 0.1)
 *
 * @example
 * const buffer = root.createBuffer(d.vec2f, d.vec2f(0, 1)); // buffer holding a d.vec2f value, with an initial value of vec2f(0, 1);
 */
export const vec2f = makeVecSchema(Vec2fImpl) as NativeVec2f;

/**
 * Type of the `d.vec2h` object/function: vector data type schema/constructor
 */
export type NativeVec2h = Vec2h & { '~exotic': Vec2h } & ((
    x: number,
    y: number,
  ) => v2h) &
  ((xy: number) => v2h) &
  (() => v2h);

/**
 *
 * Schema representing vec2h - a vector with 2 elements of type f16.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec2h(); // (0.0, 0.0)
 * const vector = d.vec2h(1); // (1.0, 1.0)
 * const vector = d.vec2h(0.5, 0.1); // (0.5, 0.1)
 *
 * @example
 * const buffer = root.createBuffer(d.vec2h, d.vec2h(0, 1)); // buffer holding a d.vec2h value, with an initial value of vec2h(0, 1);
 */
export const vec2h = makeVecSchema(Vec2hImpl) as NativeVec2h;

/**
 * Type of the `d.vec2i` object/function: vector data type schema/constructor
 */
export type NativeVec2i = Vec2i & { '~exotic': Vec2i } & ((
    x: number,
    y: number,
  ) => v2i) &
  ((xy: number) => v2i) &
  (() => v2i);

/**
 *
 * Schema representing vec2i - a vector with 2 elements of type i32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec2i(); // (0, 0)
 * const vector = d.vec2i(1); // (1, 1)
 * const vector = d.vec2i(-1, 1); // (-1, 1)
 *
 * @example
 * const buffer = root.createBuffer(d.vec2i, d.vec2i(0, 1)); // buffer holding a d.vec2i value, with an initial value of vec2i(0, 1);
 */
export const vec2i = makeVecSchema(Vec2iImpl) as NativeVec2i;

/**
 * Type of the `d.vec2u` object/function: vector data type schema/constructor
 */
export type NativeVec2u = Vec2u & { '~exotic': Vec2u } & ((
    x: number,
    y: number,
  ) => v2u) &
  ((xy: number) => v2u) &
  (() => v2u);

/**
 *
 * Schema representing vec2u - a vector with 2 elements of type u32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec2u(); // (0, 0)
 * const vector = d.vec2u(1); // (1, 1)
 * const vector = d.vec2u(1, 2); // (1, 2)
 *
 * @example
 * const buffer = root.createBuffer(d.vec2u, d.vec2u(0, 1)); // buffer holding a d.vec2u value, with an initial value of vec2u(0, 1);
 */
export const vec2u = makeVecSchema(Vec2uImpl) as NativeVec2u;

/**
 * Type of the `d.vec3f` object/function: vector data type schema/constructor
 */
export type NativeVec3f = Vec3f & { '~exotic': Vec3f } & ((
    x: number,
    y: number,
    z: number,
  ) => v3f) &
  ((xyz: number) => v3f) &
  (() => v3f);

/**
 *
 * Schema representing vec3f - a vector with 3 elements of type f32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec3f(); // (0.0, 0.0, 0.0)
 * const vector = d.vec3f(1); // (1.0, 1.0, 1.0)
 * const vector = d.vec3f(1, 2, 3.5); // (1.0, 2.0, 3.5)
 *
 * @example
 * const buffer = root.createBuffer(d.vec3f, d.vec3f(0, 1, 2)); // buffer holding a d.vec3f value, with an initial value of vec3f(0, 1, 2);
 */
export const vec3f = makeVecSchema(Vec3fImpl) as NativeVec3f;

/**
 * Type of the `d.vec3h` object/function: vector data type schema/constructor
 */
export type NativeVec3h = Vec3h & { '~exotic': Vec3h } & ((
    x: number,
    y: number,
    z: number,
  ) => v3h) &
  ((xyz: number) => v3h) &
  (() => v3h);

/**
 *
 * Schema representing vec3h - a vector with 3 elements of type f16.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec3h(); // (0.0, 0.0, 0.0)
 * const vector = d.vec3h(1); // (1.0, 1.0, 1.0)
 * const vector = d.vec3h(1, 2, 3.5); // (1.0, 2.0, 3.5)
 *
 * @example
 * const buffer = root.createBuffer(d.vec3h, d.vec3h(0, 1, 2)); // buffer holding a d.vec3h value, with an initial value of vec3h(0, 1, 2);
 */
export const vec3h = makeVecSchema(Vec3hImpl) as NativeVec3h;

/**
 * Type of the `d.vec3i` object/function: vector data type schema/constructor
 */
export type NativeVec3i = Vec3i & { '~exotic': Vec3i } & ((
    x: number,
    y: number,
    z: number,
  ) => v3i) &
  ((xyz: number) => v3i) &
  (() => v3i);

/**
 *
 * Schema representing vec3i - a vector with 3 elements of type i32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec3i(); // (0, 0, 0)
 * const vector = d.vec3i(1); // (1, 1, 1)
 * const vector = d.vec3i(1, 2, -3); // (1, 2, -3)
 *
 * @example
 * const buffer = root.createBuffer(d.vec3i, d.vec3i(0, 1, 2)); // buffer holding a d.vec3i value, with an initial value of vec3i(0, 1, 2);
 */
export const vec3i = makeVecSchema(Vec3iImpl) as NativeVec3i;

/**
 * Type of the `d.vec3u` object/function: vector data type schema/constructor
 */
export type NativeVec3u = Vec3u & { '~exotic': Vec3u } & ((
    x: number,
    y: number,
    z: number,
  ) => v3u) &
  ((xyz: number) => v3u) &
  (() => v3u);

/**
 *
 * Schema representing vec3u - a vector with 3 elements of type u32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec3u(); // (0, 0, 0)
 * const vector = d.vec3u(1); // (1, 1, 1)
 * const vector = d.vec3u(1, 2, 3); // (1, 2, 3)
 *
 * @example
 * const buffer = root.createBuffer(d.vec3u, d.vec3u(0, 1, 2)); // buffer holding a d.vec3u value, with an initial value of vec3u(0, 1, 2);
 */
export const vec3u = makeVecSchema(Vec3uImpl) as NativeVec3u;

/**
 * Type of the `d.vec4f` object/function: vector data type schema/constructor
 */
export type NativeVec4f = Vec4f & { '~exotic': Vec4f } & ((
    x: number,
    y: number,
    z: number,
    w: number,
  ) => v4f) &
  ((xyzw: number) => v4f) &
  (() => v4f);

/**
 *
 * Schema representing vec4f - a vector with 4 elements of type f32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec4f(); // (0.0, 0.0, 0.0, 0.0)
 * const vector = d.vec4f(1); // (1.0, 1.0, 1.0, 1.0)
 * const vector = d.vec4f(1, 2, 3, 4.5); // (1.0, 2.0, 3.0, 4.5)
 *
 * @example
 * const buffer = root.createBuffer(d.vec4f, d.vec4f(0, 1, 2, 3)); // buffer holding a d.vec4f value, with an initial value of vec4f(0, 1, 2, 3);
 */
export const vec4f = makeVecSchema(Vec4fImpl) as NativeVec4f;

/**
 * Type of the `d.vec4h` object/function: vector data type schema/constructor
 */
export type NativeVec4h = Vec4h & { '~exotic': Vec4h } & ((
    x: number,
    y: number,
    z: number,
    w: number,
  ) => v4h) &
  ((xyzw: number) => v4h) &
  (() => v4h);

/**
 *
 * Schema representing vec4h - a vector with 4 elements of type f16.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec4h(); // (0.0, 0.0, 0.0, 0.0)
 * const vector = d.vec4h(1); // (1.0, 1.0, 1.0, 1.0)
 * const vector = d.vec4h(1, 2, 3, 4.5); // (1.0, 2.0, 3.0, 4.5)
 *
 * @example
 * const buffer = root.createBuffer(d.vec4h, d.vec4h(0, 1, 2, 3)); // buffer holding a d.vec4h value, with an initial value of vec4h(0, 1, 2, 3);
 */
export const vec4h = makeVecSchema(Vec4hImpl) as NativeVec4h;

/**
 * Type of the `d.vec4i` object/function: vector data type schema/constructor
 */
export type NativeVec4i = Vec4i & { '~exotic': Vec4i } & ((
    x: number,
    y: number,
    z: number,
    w: number,
  ) => v4i) &
  ((xyzw: number) => v4i) &
  (() => v4i);

/**
 *
 * Schema representing vec4i - a vector with 4 elements of type i32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec4i(); // (0, 0, 0, 0)
 * const vector = d.vec4i(1); // (1, 1, 1, 1)
 * const vector = d.vec4i(1, 2, 3, -4); // (1, 2, 3, -4)
 *
 * @example
 * const buffer = root.createBuffer(d.vec4i, d.vec4i(0, 1, 2, 3)); // buffer holding a d.vec4i value, with an initial value of vec4i(0, 1, 2, 3);
 */
export const vec4i = makeVecSchema(Vec4iImpl) as NativeVec4i;

/**
 * Type of the `d.vec4u` object/function: vector data type schema/constructor
 */
export type NativeVec4u = Vec4u & { '~exotic': Vec4u } & ((
    x: number,
    y: number,
    z: number,
    w: number,
  ) => v4u) &
  ((xyzw: number) => v4u) &
  (() => v4u);

/**
 *
 * Schema representing vec4u - a vector with 4 elements of type u32.
 * Also a constructor function for this vector value.
 *
 * @example
 * const vector = d.vec4u(); // (0, 0, 0, 0)
 * const vector = d.vec4u(1); // (1, 1, 1, 1)
 * const vector = d.vec4u(1, 2, 3, 4); // (1, 2, 3, 4)
 *
 * @example
 * const buffer = root.createBuffer(d.vec4u, d.vec4u(0, 1, 2, 3)); // buffer holding a d.vec4u value, with an initial value of vec4u(0, 1, 2, 3);
 */
export const vec4u = makeVecSchema(Vec4uImpl) as NativeVec4u;
