import { LineSegmentVertex } from '@typegpu/geometry';
import { perlin2d } from '@typegpu/noise';
import tgpu from 'typegpu';
import { arrayOf, f32, i32, mat2x2f, u32, vec2f } from 'typegpu/data';
import {
  add,
  clamp,
  cos,
  floor,
  max,
  mul,
  pow,
  select,
  sin,
} from 'typegpu/std';
import { TEST_SEGMENT_COUNT } from './constants.ts';
import {
  randFloat01,
  randSeed,
} from '../../../../../../../packages/typegpu-noise/src/random.ts';

const testCaseShell = tgpu.fn([u32, f32], LineSegmentVertex);

const segmentSide = tgpu['~unstable'].const(arrayOf(f32, 4), [-1, -1, 1, 1]);

export const segmentAlternate = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const side = segmentSide.$[vertexIndex];
    const r = sin(t + select(0, Math.PI / 2, side === -1));
    const radius = 0.4 * r * r;
    return LineSegmentVertex({
      position: vec2f(0.5 * side * cos(t), 0.5 * side * sin(t)),
      radius,
    });
  },
);

export const segmentStretch = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const side = segmentSide.$[vertexIndex];
    const distance = 0.5 * clamp(0.55 * sin(1.5 * t) + 0.5, 0, 1);
    return LineSegmentVertex({
      position: vec2f(distance * side * cos(t), distance * side * sin(t)),
      radius: 0.25,
    });
  },
);

export const segmentContainsAnotherEnd = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const side = segmentSide.$[vertexIndex];
    return LineSegmentVertex({
      position: vec2f(side * 0.25 * (1 + clamp(sin(t), -0.8, 1)), 0),
      radius: 0.25 + side * 0.125,
    });
  },
);

export const caseVShapeSmall = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const side = clamp(f32(vertexIndex) - 2, -1, 1);
    const isMiddle = side === 0;
    return LineSegmentVertex({
      position: vec2f(0.5 * side, select(0.5 * cos(t), 0, isMiddle)),
      radius: select(0.1, 0.2, isMiddle),
    });
  },
);

export const caseVShapeBig = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const side = clamp(f32(vertexIndex) - 2, -1, 1);
    const isMiddle = side === 0;
    return LineSegmentVertex({
      position: vec2f(0.5 * side, select(0.5 * cos(t), 0, isMiddle)),
      radius: select(0.3, 0.2, isMiddle),
    });
  },
);

export const halfCircle = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const angle = Math.PI * clamp(f32(vertexIndex) - 1, 0, 50) / 50;
    const radius = 0.5 * cos(t);
    return LineSegmentVertex({
      position: vec2f(radius * cos(angle), radius * sin(angle)),
      radius: 0.2,
    });
  },
);

export const bending = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const i = clamp(f32(vertexIndex) - 1, 0, 48) / 48;
    const x = 2 * i - 1;
    const s = sin(t);
    const n = 10 * s * s * s * s + 0.25;
    return LineSegmentVertex({
      position: vec2f(0.5 * x, 0.5 * pow(1 - pow(x, n), 1 / n)),
      radius: 0.2,
    });
  },
);

export const animateWidth = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const i = (f32(vertexIndex) % TEST_SEGMENT_COUNT) / TEST_SEGMENT_COUNT;
    const x = cos(4 * 2 * Math.PI * i + Math.PI / 2);
    const y = cos(5 * 2 * Math.PI * i);
    return LineSegmentVertex({
      position: vec2f(0.8 * x, 0.8 * y),
      radius: 0.05 * clamp(sin(8 * Math.PI * i - 3 * t), 0.1, 1),
    });
  },
);

export const perlinTraces = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const perLine = u32(200);
    const i = f32(max(vertexIndex, 0)) / f32(perLine);
    const n = floor(i);
    const x = 2 * (i - n) - 1;
    const value = 0.5 * perlin2d.sample(vec2f(2 * x + 2 * t, t + 0.1 * n)) +
      0.25 * perlin2d.sample(vec2f(4 * x, t + 100 + 0.1 * n)) +
      0.125 * perlin2d.sample(vec2f(8 * x, t + 200 + 0.2 * n)) +
      0.0625 * perlin2d.sample(vec2f(16 * x, t + 300 + 0.3 * n));
    const y = 0.125 * n - 0.5 + 0.5 * value;
    const radiusFactor = 0.025 * (n + 1);
    return LineSegmentVertex({
      position: vec2f(0.8 * x, y),
      radius: select(
        radiusFactor * radiusFactor,
        -1,
        vertexIndex % perLine === 0,
      ),
    });
  },
);

export const arms = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const s = sin(t);
    const c = cos(t);
    const r = 0.25;
    const points = [
      vec2f(r * s - 0.25, r * c),
      vec2f(-0.25, 0),
      vec2f(0.25, 0),
      vec2f(-r * s + 0.25, r * c),
    ];
    const i = clamp(i32(vertexIndex) - 1, 0, 3);
    return LineSegmentVertex({
      position: points[i],
      radius: 0.2,
    });
  },
);

export const aarmsSmall = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const result = arms(vertexIndex, t);
    return LineSegmentVertex({
      position: result.position,
      radius: select(0.1, 0.2, vertexIndex === 2 || vertexIndex === 3),
    });
  },
);

export const armsBig = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const result = arms(vertexIndex, t);
    return LineSegmentVertex({
      position: result.position,
      radius: select(0.275, 0.1, vertexIndex === 2 || vertexIndex === 3),
    });
  },
);

export const armsRotating = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const s = sin(t);
    const c = cos(t);
    const r = 0.25;
    const points = [
      vec2f(r * s - 0.25, r * c),
      vec2f(-0.25, 0),
      vec2f(0.25, 0),
      vec2f(-r * s + 0.25, -r * c),
    ];
    const i = clamp(i32(vertexIndex) - 1, 0, 3);
    return LineSegmentVertex({
      position: points[i],
      radius: 0.2,
    });
  },
);

export const flyingSquares = testCaseShell(
  (vertexIndex, t) => {
    'kernel';
    const squareIndex = vertexIndex / 8;
    randSeed(f32(squareIndex + 5));
    const squarePoints = [
      vec2f(-1, -1),
      vec2f(1, -1),
      vec2f(1, 1),
      vec2f(-1, 1),
    ];
    const pointIndex = vertexIndex % 8;
    const point = squarePoints[pointIndex % 4];
    const rotationSpeed = 2 * randFloat01() - 1;
    const s = sin(t * rotationSpeed);
    const c = cos(t * rotationSpeed);
    const rotate = mat2x2f(c, -s, s, c);
    const r = 0.1 + 0.05 * randFloat01();
    // TODO: why do I need to wrap with f32?
    const x = f32(2.0 * randFloat01() - 1);
    const y = f32(2.0 * randFloat01() - 1);
    const transformedPoint = add(vec2f(x, y), mul(rotate, mul(point, r)));
    return LineSegmentVertex({
      position: transformedPoint,
      radius: select(
        0.1 * r + 0.05 * randFloat01(),
        -1,
        pointIndex === 7 || squareIndex > 50,
      ),
    });
  },
);
