/**
 * @vitest-environment jsdom
 */

import { describe, expect } from 'vitest';
import { it } from 'typegpu-testing-utility';
import { runExampleTest, setupCommonMocks } from './utils/baseTest.ts';
import { mockResizeObserver } from './utils/commonMocks.ts';

describe('subdivided-triangles example', () => {
  setupCommonMocks();

  it('should produce valid code', async ({ device }) => {
    const shaderCodes = await runExampleTest(
      {
        category: 'geometry',
        name: 'subdivided-triangles',
        setupMocks: mockResizeObserver,
        expectedCalls: 1,
      },
      device,
    );

    expect(shaderCodes).toMatchInlineSnapshot(`
      "struct Uniforms {
        maxSubdivCount: u32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct Triangle {
        a: vec2f,
        b: vec2f,
        c: vec2f,
      }

      @group(0) @binding(1) var<storage, read> triangles: array<Triangle>;

      fn subdivTriangle(A: vec2f, B: vec2f, C: vec2f, vertexIndex: u32, maxSubdivCount: u32) -> vec2f {
        let level = u32(((sqrt(((8f * f32(vertexIndex)) + 1f)) - 1f) / 2f));
        let startIndex = ((level * (level + 1u)) >> 1u);
        let j = (vertexIndex - startIndex);
        let i = (level - j);
        let ti = (f32(i) / f32(maxSubdivCount));
        let tj = (f32(j) / f32(maxSubdivCount));
        return ((mix(A, B, ti) + mix(A, C, tj)) - A);
      }

      struct mainVertex_Output {
        @builtin(position) outPos: vec4f,
        @location(0) @interpolate(flat) instanceIndex: u32,
      }

      @vertex fn mainVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> mainVertex_Output {
        let maxSubdivCount = uniforms.maxSubdivCount;
        let T = (&triangles[instanceIndex]);
        let pos = subdivTriangle((*T).a, (*T).b, (*T).c, vertexIndex, maxSubdivCount);
        return mainVertex_Output(vec4f(pos, 0f, 1f), instanceIndex);
      }

      struct mainFragment_Input {
        @location(0) @interpolate(flat) instanceIndex: u32,
      }

      @fragment fn mainFragment(_arg_0: mainFragment_Input) -> @location(0) vec4f {
        let color = vec3f(1f, cos(f32(_arg_0.instanceIndex)), sin((5f * f32(_arg_0.instanceIndex))));
        return vec4f(color, 1f);
      }

      struct Uniforms {
        maxSubdivCount: u32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct Triangle {
        a: vec2f,
        b: vec2f,
        c: vec2f,
      }

      @group(0) @binding(1) var<storage, read> triangles: array<Triangle>;

      fn subdivTriangle(A: vec2f, B: vec2f, C: vec2f, vertexIndex: u32, maxSubdivCount: u32) -> vec2f {
        let level = u32(((sqrt(((8f * f32(vertexIndex)) + 1f)) - 1f) / 2f));
        let startIndex = ((level * (level + 1u)) >> 1u);
        let j = (vertexIndex - startIndex);
        let i = (level - j);
        let ti = (f32(i) / f32(maxSubdivCount));
        let tj = (f32(j) / f32(maxSubdivCount));
        return ((mix(A, B, ti) + mix(A, C, tj)) - A);
      }

      struct mainVertex_Output {
        @builtin(position) outPos: vec4f,
        @location(0) @interpolate(flat) instanceIndex: u32,
      }

      @vertex fn mainVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> mainVertex_Output {
        let maxSubdivCount = uniforms.maxSubdivCount;
        let T = (&triangles[instanceIndex]);
        let pos = subdivTriangle((*T).a, (*T).b, (*T).c, vertexIndex, maxSubdivCount);
        return mainVertex_Output(vec4f(pos, 0f, 1f), instanceIndex);
      }

      @fragment fn wireframeFragment() -> @location(0) vec4f {
        return vec4f(1);
      }"
    `);
  });
});
