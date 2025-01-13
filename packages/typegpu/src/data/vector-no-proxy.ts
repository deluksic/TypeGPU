import { vectorComponentCombinations } from '../shared/generators';
import type { SelfResolvable } from '../types';

/**
 * Type encompassing all available kinds of vector.
 */
export type VecKind =
  | 'vec2f'
  | 'vec2i'
  | 'vec2u'
  | 'vec2h'
  | 'vec3f'
  | 'vec3i'
  | 'vec3u'
  | 'vec3h'
  | 'vec4f'
  | 'vec4i'
  | 'vec4u'
  | 'vec4h';

abstract class VecBase extends Array implements SelfResolvable {
  declare kind: VecKind;

  '~resolve'(): string {
    return `${this.kind}(${this.join(', ')})`;
  }

  toString() {
    return this['~resolve']();
  }
}

type Tuple2 = [number, number];
type Tuple3 = [number, number, number];
type Tuple4 = [number, number, number, number];

abstract class Vec2 extends VecBase implements Tuple2 {
  declare readonly length = 2;

  0: number;
  1: number;

  constructor(x?: number, y?: number) {
    super();
    this[0] = x ?? 0;
    this[1] = y ?? x ?? 0;
  }

  get x() {
    return this[0];
  }

  get y() {
    return this[1];
  }

  set x(value: number) {
    this[0] = value;
  }

  set y(value: number) {
    this[1] = value;
  }
}

abstract class Vec3 extends VecBase implements Tuple3 {
  declare readonly length = 3;

  0: number;
  1: number;
  2: number;

  constructor(x?: number, y?: number, z?: number) {
    super();
    this[0] = x ?? 0;
    this[1] = y ?? x ?? 0;
    this[2] = z ?? x ?? 0;
  }

  get x() {
    return this[0];
  }

  get y() {
    return this[1];
  }

  get z() {
    return this[2];
  }

  set x(value: number) {
    this[0] = value;
  }

  set y(value: number) {
    this[1] = value;
  }

  set z(value: number) {
    this[2] = value;
  }
}

abstract class Vec4 extends VecBase implements Tuple4 {
  declare readonly length = 4;

  0: number;
  1: number;
  2: number;
  3: number;

  constructor(x?: number, y?: number, z?: number, w?: number) {
    super();
    this[0] = x ?? 0;
    this[1] = y ?? x ?? 0;
    this[2] = z ?? x ?? 0;
    this[3] = w ?? x ?? 0;
  }

  get x() {
    return this[0];
  }

  get y() {
    return this[1];
  }

  get z() {
    return this[2];
  }

  get w() {
    return this[3];
  }

  set x(value: number) {
    this[0] = value;
  }

  set y(value: number) {
    this[1] = value;
  }

  set z(value: number) {
    this[2] = value;
  }

  set w(value: number) {
    this[3] = value;
  }
}

export class Vec2fImpl extends Vec2 {
  readonly kind = 'vec2f';
}

export class Vec2hImpl extends Vec2 {
  readonly kind = 'vec2h';
}

export class Vec2iImpl extends Vec2 {
  readonly kind = 'vec2i';
}

export class Vec2uImpl extends Vec2 {
  readonly kind = 'vec2u';
}

export class Vec3fImpl extends Vec3 {
  readonly kind = 'vec3f';
}

export class Vec3hImpl extends Vec3 {
  readonly kind = 'vec3h';
}

export class Vec3iImpl extends Vec3 {
  readonly kind = 'vec3i';
}

export class Vec3uImpl extends Vec3 {
  readonly kind = 'vec3u';
}

export class Vec4fImpl extends Vec4 {
  readonly kind = 'vec4f';
}

export class Vec4hImpl extends Vec4 {
  readonly kind = 'vec4h';
}

export class Vec4iImpl extends Vec4 {
  readonly kind = 'vec4i';
}

export class Vec4uImpl extends Vec4 {
  readonly kind = 'vec4u';
}

/**
 * Adds swizzling combinations to the target class prototype, by constructing
 * getters using `new Function()`. This should be faster than doing it using
 * Proxy because the generated functions are static and very small, no runtime if / switch.
 * Example implementation of such generated getter:
 * ```
 * get xxyw() {
 *   return new this._Vec4(this[0], this[0], this[1], this[3]);
 * }
 * ```
 */
function enableSwizzlingFor(
  TargetClass: typeof VecBase,
  components: string,
  vecs: Record<number, typeof VecBase>,
) {
  const componentIndex: Record<string, number> = { x: 0, y: 1, z: 2, w: 3 };
  for (const count of [2, 3, 4] as const) {
    const VecClass = vecs[count];
    const vecClassName = `_Vec${count}`;
    Object.defineProperty(TargetClass.prototype, vecClassName, {
      value: VecClass,
      configurable: false,
      enumerable: false,
      writable: false,
    });
    for (const swizzle of vectorComponentCombinations(components, count)) {
      const getImplementation = new Function(
        `return new this.${vecClassName}(
          ${[...swizzle].map((s) => `this[${componentIndex[s]}]`)}
        )`,
      ) as () => unknown;
      // Add a getter to the class prototype
      Object.defineProperty(TargetClass.prototype, swizzle, {
        get: getImplementation,
        configurable: false,
        enumerable: false,
      });
    }
  }
}

export function enableSwizzling() {
  const vecType = {
    f: { 2: Vec2fImpl, 3: Vec3fImpl, 4: Vec4fImpl },
    h: { 2: Vec2hImpl, 3: Vec3hImpl, 4: Vec4hImpl },
    i: { 2: Vec2iImpl, 3: Vec3iImpl, 4: Vec4iImpl },
    u: { 2: Vec2uImpl, 3: Vec3uImpl, 4: Vec4uImpl },
  };

  enableSwizzlingFor(Vec2fImpl, 'xy', vecType.f);
  enableSwizzlingFor(Vec2hImpl, 'xy', vecType.h);
  enableSwizzlingFor(Vec2iImpl, 'xy', vecType.i);
  enableSwizzlingFor(Vec2uImpl, 'xy', vecType.u);

  enableSwizzlingFor(Vec3fImpl, 'xyz', vecType.f);
  enableSwizzlingFor(Vec3hImpl, 'xyz', vecType.h);
  enableSwizzlingFor(Vec3iImpl, 'xyz', vecType.i);
  enableSwizzlingFor(Vec3uImpl, 'xyz', vecType.u);

  enableSwizzlingFor(Vec4fImpl, 'xyzw', vecType.f);
  enableSwizzlingFor(Vec4hImpl, 'xyzw', vecType.h);
  enableSwizzlingFor(Vec4iImpl, 'xyzw', vecType.i);
  enableSwizzlingFor(Vec4uImpl, 'xyzw', vecType.u);
}

enableSwizzling();
