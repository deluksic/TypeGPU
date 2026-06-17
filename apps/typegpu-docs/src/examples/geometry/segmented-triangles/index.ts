import {
  segmentTriangle,
  segmentTriangleIndices,
  segmentTriangleWireframeIndices,
} from '@typegpu/geometry';
import tgpu, { d, std as s } from 'typegpu';
import { defineControls } from '../../common/defineControls.ts';

const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const canvas = document.querySelector('canvas');
const context = canvas?.getContext('webgpu');
const multisample = true;

if (!canvas) {
  throw new Error('Could not find canvas');
}
if (!context) {
  throw new Error('Could not create WebGPU context');
}

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter?.requestDevice({
  requiredFeatures: ['timestamp-query'],
});
if (!device) {
  throw new Error('Could not get WebGPU device');
}
const root = tgpu.initFromDevice({ device });

context.configure({
  device: root.device,
  format: presentationFormat,
  alphaMode: 'premultiplied',
});

let msaaTexture: GPUTexture;
let msaaTextureView: GPUTextureView;

const createDepthAndMsaaTextures = () => {
  if (msaaTexture) {
    msaaTexture.destroy();
  }
  msaaTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: presentationFormat,
    sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  msaaTextureView = msaaTexture.createView();
};

createDepthAndMsaaTextures();
const resizeObserver = new ResizeObserver(createDepthAndMsaaTextures);
resizeObserver.observe(canvas);

const MAX_SEGMENT_COUNT = 10;

const Uniforms = d.struct({
  maxSegmentCount: d.u32,
});

const Triangle = d.struct({
  a: d.vec2f,
  b: d.vec2f,
  c: d.vec2f,
});

const bindGroupLayout = tgpu.bindGroupLayout({
  uniforms: {
    uniform: Uniforms,
  },
  triangles: {
    storage: (n: number) => d.arrayOf(Triangle, n),
  },
});

const maxSegmentIndices = segmentTriangleIndices(MAX_SEGMENT_COUNT);
const maxWireframeIndices = segmentTriangleWireframeIndices(MAX_SEGMENT_COUNT);

const indexBuffer = root
  .createBuffer(d.arrayOf(d.u32, maxSegmentIndices.length), maxSegmentIndices)
  .$usage('index');

const wireframeIndexBuffer = root
  .createBuffer(d.arrayOf(d.u32, maxWireframeIndices.length), maxWireframeIndices)
  .$usage('index');

const triangleCount = 120;
const twoPi = Math.PI * 2;

const triangles = root
  .createBuffer(
    d.arrayOf(Triangle, triangleCount),
    Array.from({ length: triangleCount }).map(() => {
      const cx = Math.random() * 1.4 - 0.7;
      const cy = Math.random() * 1.4 - 0.7;
      const size = 0.12 + Math.random() * 0.1;
      const rotation = Math.random() * twoPi;

      const point = (offset: number) =>
        d.vec2f(cx + size * Math.cos(rotation + offset), cy + size * Math.sin(rotation + offset));

      return Triangle({
        a: point(0),
        b: point(twoPi / 3),
        c: point((2 * twoPi) / 3),
      });
    }),
  )
  .$usage('storage');

let segmentCount = 8;

const uniforms = root.createBuffer(Uniforms, { maxSegmentCount: segmentCount }).$usage('uniform');

const uniformsBindGroup = root.createBindGroup(bindGroupLayout, {
  uniforms,
  triangles,
});

const mainVertex = tgpu.vertexFn({
  in: {
    vertexIndex: d.builtin.vertexIndex,
    instanceIndex: d.builtin.instanceIndex,
  },
  out: {
    outPos: d.builtin.position,
    instanceIndex: d.interpolate('flat', d.u32),
  },
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const maxSegmentCount = bindGroupLayout.$.uniforms.maxSegmentCount;
  const T = bindGroupLayout.$.triangles[instanceIndex];
  const pos = segmentTriangle(T.a, T.b, T.c, vertexIndex, maxSegmentCount);
  return {
    outPos: d.vec4f(pos, 0.0, 1.0),
    instanceIndex,
  };
});

const mainFragment = tgpu.fragmentFn({
  in: {
    instanceIndex: d.interpolate('flat', d.u32),
  },
  out: d.vec4f,
})(({ instanceIndex }) => {
  'use gpu';
  const color = d.vec3f(1, s.cos(d.f32(instanceIndex)), s.sin(5 * d.f32(instanceIndex)));
  return d.vec4f(color, 1);
});

const wireframeFragment = tgpu.fragmentFn({
  out: d.vec4f,
})(() => {
  'use gpu';
  return d.vec4f(1, 1, 1, 1);
});

const fillPipeline = root
  .createRenderPipeline({
    vertex: mainVertex,
    fragment: mainFragment,
    targets: { format: presentationFormat },
    multisample: { count: multisample ? 4 : 1 },
  })
  .withIndexBuffer(indexBuffer);

const wireframePipeline = root
  .createRenderPipeline({
    vertex: mainVertex,
    fragment: wireframeFragment,
    targets: { format: presentationFormat },
    primitive: {
      topology: 'line-list',
    },
    multisample: { count: multisample ? 4 : 1 },
  })
  .withIndexBuffer(wireframeIndexBuffer);

function indexCountForSegmentCount(level: number) {
  return level * level * 3;
}

function wireframeIndexCountForSegmentCount(level: number) {
  return level * level * 6;
}

function colorAttachment(clear: boolean) {
  return {
    ...(multisample
      ? {
          view: msaaTextureView,
          resolveTarget: context,
        }
      : { view: context }),
    clearValue: [0, 0, 0, 0],
    loadOp: clear ? ('clear' as const) : ('load' as const),
    storeOp: 'store' as const,
  };
}

function draw() {
  fillPipeline
    .with(uniformsBindGroup)
    .withColorAttachment(colorAttachment(true))
    .withPerformanceCallback((a, b) => {
      console.log((Number(b - a) * 1e-6).toFixed(3), 'ms');
    })
    .drawIndexed(indexCountForSegmentCount(segmentCount), triangleCount);

  wireframePipeline
    .with(uniformsBindGroup)
    .withColorAttachment(colorAttachment(false))
    .drawIndexed(wireframeIndexCountForSegmentCount(segmentCount), triangleCount);
}

setTimeout(draw, 100);

// #region Example controls & Cleanup

export const controls = defineControls({
  Segments: {
    initial: segmentCount,
    min: 1,
    max: MAX_SEGMENT_COUNT,
    step: 1,
    onSliderChange: (newValue) => {
      segmentCount = newValue;
      uniforms.write({ maxSegmentCount: segmentCount });
      draw();
    },
  },
});

export function onCleanup() {
  root.destroy();
}

// #endregion
