/*
{
  "title": "Gradient Tiles"
}
*/

import {
  addElement,
  addParameter,
  onCleanup,
  onFrame,
} from '@wigsill/example-toolkit';
import {
  createRuntime,
  makeArena,
  struct,
  u32,
  vec2f,
  vec4f,
  wgsl,
} from 'wigsill';

const runtime = await createRuntime();
const device = runtime.device;

const xSpanData = wgsl.buffer(u32).$name('x-span');
const ySpanData = wgsl.buffer(u32).$name('y-span');

const mainArena = makeArena({
  bufferBindingType: 'uniform',
  memoryEntries: [xSpanData, ySpanData],
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
});

const canvas = await addElement('canvas');
const context = canvas.getContext('webgpu') as GPUCanvasContext;

const devicePixelRatio = window.devicePixelRatio;
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

context.configure({
  device,
  format: presentationFormat,
  alphaMode: 'premultiplied',
});

const outputStruct = struct({
  '@builtin(position) pos': vec4f,
  '@location(0) uv': vec2f,
});

const renderPipeline = runtime.makeRenderPipeline({
  vertex: {
    args: ['@builtin(vertex_index) VertexIndex: u32'],
    output: outputStruct,
    code: wgsl`
      var pos = array<vec2f, 4>(
        vec2(0.5, 0.5), // top-right
        vec2(-0.5, 0.5), // top-left
        vec2(0.5, -0.5), // bottom-right
        vec2(-0.5, -0.5) // bottom-left
      );

      var uv = array<vec2f, 4>(
        vec2(1., 1.), // top-right
        vec2(0., 1.), // top-left
        vec2(1., 0.), // bottom-right
        vec2(0., 0.) // bottom-left
      );

      var output: ${outputStruct};
      output.pos = vec4f(pos[VertexIndex], 0.0, 1.0);
      output.uv = uv[VertexIndex];
      return output;
    `,
  },

  fragment: {
    args: ['@builtin(position) Position: vec4f', '@location(0) uv: vec2f'],
    code: wgsl.code`
      let red = floor(uv.x * f32(${xSpanData})) / f32(${xSpanData});
      let green = floor(uv.y * f32(${ySpanData})) / f32(${ySpanData});
      return vec4(red, green, 0.5, 1.0);
    `,
    output: '@location(0) vec4f',
    target: [
      {
        format: presentationFormat,
      },
    ],
  },

  primitive: {
    topology: 'triangle-strip',
  },

  arenas: [mainArena],
});

addParameter(
  'x-span',
  { initial: 16, min: 1, max: 16, step: 1 },
  (xSpan: number) => xSpanData.write(runtime, xSpan),
);

addParameter(
  'y-span',
  { initial: 16, min: 1, max: 16, step: 1 },
  (ySpan: number) => ySpanData.write(runtime, ySpan),
);

onFrame(() => {
  const textureView = context.getCurrentTexture().createView();

  renderPipeline.execute({
    colorAttachments: [
      {
        view: textureView,
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],

    vertexCount: 4,
  });

  runtime.flush();
});

onCleanup(() => {
  // TODO: Clean up
});
