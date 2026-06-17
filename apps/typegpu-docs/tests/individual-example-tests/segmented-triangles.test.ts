/**
 * @vitest-environment jsdom
 */

import { describe, expect } from 'vitest';
import { it } from 'typegpu-testing-utility';
import { runExampleTest, setupCommonMocks } from './utils/baseTest.ts';
import { mockResizeObserver } from './utils/commonMocks.ts';

describe('segmented-triangles example', () => {
  setupCommonMocks();

  it('should produce valid code', async ({ device }) => {
    const shaderCodes = await runExampleTest(
      {
        category: 'geometry',
        name: 'segmented-triangles',
        setupMocks: mockResizeObserver,
        expectedCalls: 1,
      },
      device,
    );

    expect(shaderCodes).toMatchInlineSnapshot(`
      "struct Uniforms {
        maxSegmentCount: u32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct Triangle {
        a: vec2f,
        b: vec2f,
        c: vec2f,
      }

      @group(0) @binding(1) var<storage, read> triangles: array<Triangle>;

      fn triangleGridBarycentrics(vertexIndex: u32, maxSegmentCount: u32) -> vec3f {
        let level = u32(((sqrt(((8f * f32(vertexIndex)) + 1f)) - 1f) / 2f));
        let startIndex = ((level * (level + 1u)) >> 1u);
        let j = f32((vertexIndex - startIndex));
        let i = (f32(level) - j);
        let invN = (1f / f32(maxSegmentCount));
        return vec3f((1f - ((i + j) * invN)), (i * invN), (j * invN));
      }

      fn segmentTriangle(A: vec2f, B: vec2f, C: vec2f, vertexIndex: u32, maxSegmentCount: u32) -> vec2f {
        let w = triangleGridBarycentrics(vertexIndex, maxSegmentCount);
        return (((A * w.x) + (B * w.y)) + (C * w.z));
      }

      struct mainVertex_Output {
        @builtin(position) outPos: vec4f,
        @location(0) @interpolate(flat) instanceIndex: u32,
      }

      @vertex fn mainVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> mainVertex_Output {
        let maxSegmentCount = uniforms.maxSegmentCount;
        let T = (&triangles[instanceIndex]);
        let pos = segmentTriangle((*T).a, (*T).b, (*T).c, vertexIndex, maxSegmentCount);
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
        maxSegmentCount: u32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;

      struct Triangle {
        a: vec2f,
        b: vec2f,
        c: vec2f,
      }

      @group(0) @binding(1) var<storage, read> triangles: array<Triangle>;

      fn triangleGridBarycentrics(vertexIndex: u32, maxSegmentCount: u32) -> vec3f {
        let level = u32(((sqrt(((8f * f32(vertexIndex)) + 1f)) - 1f) / 2f));
        let startIndex = ((level * (level + 1u)) >> 1u);
        let j = f32((vertexIndex - startIndex));
        let i = (f32(level) - j);
        let invN = (1f / f32(maxSegmentCount));
        return vec3f((1f - ((i + j) * invN)), (i * invN), (j * invN));
      }

      fn segmentTriangle(A: vec2f, B: vec2f, C: vec2f, vertexIndex: u32, maxSegmentCount: u32) -> vec2f {
        let w = triangleGridBarycentrics(vertexIndex, maxSegmentCount);
        return (((A * w.x) + (B * w.y)) + (C * w.z));
      }

      struct mainVertex_Output {
        @builtin(position) outPos: vec4f,
        @location(0) @interpolate(flat) instanceIndex: u32,
      }

      @vertex fn mainVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> mainVertex_Output {
        let maxSegmentCount = uniforms.maxSegmentCount;
        let T = (&triangles[instanceIndex]);
        let pos = segmentTriangle((*T).a, (*T).b, (*T).c, vertexIndex, maxSegmentCount);
        return mainVertex_Output(vec4f(pos, 0f, 1f), instanceIndex);
      }

      @fragment fn wireframeFragment() -> @location(0) vec4f {
        return vec4f(1);
      }"
    `);
  });
});
