import {
  roundedBox,
  ROUNDED_BOX_CORNER_COUNT,
  ROUNDED_BOX_EDGE_PATCH_OFFSET,
  ROUNDED_BOX_PATCH_COUNT,
  roundedBoxInstanceCount,
  roundedBoxIndexCountPerPatch,
  roundedBoxWireframeIndexCountPerPatch,
  subdivTriangleIndices,
  subdivTriangleWireframeIndices,
} from '@typegpu/geometry';
import tgpu, { d, std as s } from 'typegpu';
import { Camera, setupOrbitCamera } from '../../common/setup-orbit-camera.ts';
import { defineControls } from '../../common/defineControls.ts';

const root = await tgpu.init();
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const multisample = true;

const MAX_SUBDIV = 16;

const Uniforms = d.struct({
  subdivCount: d.u32,
});

const RoundedBox = d.struct({
  position: d.vec3f,
  halfSize: d.vec3f,
  cornerRadius: d.f32,
});

const bindGroupLayout = tgpu.bindGroupLayout({
  uniforms: {
    uniform: Uniforms,
  },
  camera: {
    uniform: Camera,
  },
  boxes: {
    storage: (n: number) => d.arrayOf(RoundedBox, n),
  },
});

const indexBuffer = root
  .createBuffer(
    d.arrayOf(d.u32, subdivTriangleIndices(MAX_SUBDIV).length),
    subdivTriangleIndices(MAX_SUBDIV),
  )
  .$usage('index');

const wireframeIndexBuffer = root
  .createBuffer(
    d.arrayOf(d.u32, subdivTriangleWireframeIndices(MAX_SUBDIV).length),
    subdivTriangleWireframeIndices(MAX_SUBDIV),
  )
  .$usage('index');

const boxCount = 1;

const halfSize = d.vec3f(2.5, 2.5, 2.5);
let cornerRadius = 0.6;

const boxes = root
  .createBuffer(
    d.arrayOf(RoundedBox, boxCount),
    [
      RoundedBox({
        position: d.vec3f(0, 0, 0),
        halfSize,
        cornerRadius,
      }),
    ],
  )
  .$usage('storage');

let subdivLevel = 4;

const instanceCount = roundedBoxInstanceCount(boxCount);

const uniforms = root.createBuffer(Uniforms, { subdivCount: subdivLevel }).$usage('uniform');
const camera = root.createBuffer(Camera).$usage('uniform');

const bindGroup = root.createBindGroup(bindGroupLayout, {
  uniforms,
  camera,
  boxes,
});

const { cleanupCamera } = setupOrbitCamera(
  canvas,
  {
    initPos: d.vec4f(0, 4, 12, 1),
    target: d.vec4f(0, 0, 0, 1),
    minZoom: 4,
    maxZoom: 40,
  },
  (updates) => camera.patch(updates),
);

const mainVertex = tgpu.vertexFn({
  in: {
    vertexIndex: d.builtin.vertexIndex,
    instanceIndex: d.builtin.instanceIndex,
  },
  out: {
    outPos: d.builtin.position,
    worldNormal: d.vec3f,
    patchIndex: d.interpolate('flat', d.u32),
  },
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const subdivCount = bindGroupLayout.$.uniforms.subdivCount;
  const objectIndex = d.u32(instanceIndex / ROUNDED_BOX_PATCH_COUNT);
  const boxData = bindGroupLayout.$.boxes[objectIndex];
  const patchIndex = d.u32(instanceIndex % ROUNDED_BOX_PATCH_COUNT);
  const patch = roundedBox(
    instanceIndex,
    vertexIndex,
    subdivCount,
    boxData.halfSize,
    boxData.cornerRadius,
  );
  const worldPos = boxData.position + patch.vertex;
  const cameraUniform = bindGroupLayout.$.camera;
  return {
    outPos: cameraUniform.projection * cameraUniform.view * d.vec4f(worldPos, 1),
    worldNormal: patch.normal,
    patchIndex,
  };
});

/** Patch color key (debug):
 * - Corners 0–7: octant bits (x,y,z) = (+++), (-++), (+-+), …
 * - Faces 8–19: flat mid-gray
 * - Edges 20–43: edgeIndex 0–11, two tri patches each
 *   - 0–3: parallel to Z
 *   - 4–7: parallel to X
 *   - 8–11: parallel to Y
 */
const patchBaseColor = tgpu.fn([d.u32], d.vec3f)((patchIndex) => {
  'use gpu';
  if (patchIndex < ROUNDED_BOX_CORNER_COUNT) {
    const octant = patchIndex;
    const hue = d.f32(octant) * d.f32(0.785398);
    return d.vec3f(
      0.55 + 0.25 * s.cos(hue),
      0.55 + 0.25 * s.cos(hue + d.f32(2.094)),
      0.55 + 0.25 * s.cos(hue + d.f32(4.189)),
    );
  }
  if (patchIndex < ROUNDED_BOX_EDGE_PATCH_OFFSET) {
    return d.vec3f(0.42, 0.42, 0.46);
  }
  const edgeIndex = (patchIndex - ROUNDED_BOX_EDGE_PATCH_OFFSET) >> d.u32(1);
  const hue = d.f32(edgeIndex) * d.f32(2.399963);
  return d.vec3f(
    0.45 + 0.55 * s.cos(hue),
    0.45 + 0.55 * s.cos(hue + d.f32(2.094)),
    0.45 + 0.55 * s.cos(hue + d.f32(4.189)),
  );
});

const mainFragment = tgpu.fragmentFn({
  in: {
    worldNormal: d.vec3f,
    patchIndex: d.interpolate('flat', d.u32),
  },
  out: d.vec4f,
})(({ worldNormal, patchIndex }) => {
  'use gpu';
  const lightDir = s.normalize(d.vec3f(0.4, 1, 0.3));
  const diffuse = s.max(0, s.dot(worldNormal, lightDir));
  const ambient = 0.35;
  const color = patchBaseColor(patchIndex);
  return d.vec4f(color * (ambient + diffuse * 0.85), 1);
});

const wireframeFragment = tgpu.fragmentFn({
  out: d.vec4f,
})(() => {
  'use gpu';
  return d.vec4f(1, 1, 1, 1);
});

const depthStencil = {
  format: 'depth24plus' as const,
  depthWriteEnabled: true,
  depthCompare: 'less' as const,
};

const fillPipeline = root
  .createRenderPipeline({
    vertex: mainVertex,
    fragment: mainFragment,
    targets: { format: presentationFormat },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
    depthStencil: {
      ...depthStencil,
      depthBias: 1,
      depthBiasSlopeScale: 1,
      depthBiasClamp: 0,
    },
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
      cullMode: 'none',
    },
    depthStencil: {
      ...depthStencil,
      depthWriteEnabled: false,
    },
    multisample: { count: multisample ? 4 : 1 },
  })
  .withIndexBuffer(wireframeIndexBuffer);

let msaaTexture: GPUTexture;
let msaaTextureView: GPUTextureView;
let depthTexture: GPUTexture;
let depthTextureView: GPUTextureView;

const createRenderTargets = () => {
  if (msaaTexture) {
    msaaTexture.destroy();
  }
  if (depthTexture) {
    depthTexture.destroy();
  }

  const sampleCount = multisample ? 4 : 1;

  msaaTexture = root.device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: presentationFormat,
    sampleCount,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  msaaTextureView = msaaTexture.createView();

  depthTexture = root.device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'depth24plus',
    sampleCount,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  depthTextureView = depthTexture.createView();
};

createRenderTargets();

const resizeObserver = new ResizeObserver(createRenderTargets);
resizeObserver.observe(canvas);

function colorAttachment(clear: boolean) {
  return {
    ...(multisample
      ? {
          view: msaaTextureView,
          resolveTarget: context,
        }
      : { view: context }),
    clearValue: [0.05, 0.05, 0.08, 1] as [number, number, number, number],
    loadOp: clear ? ('clear' as const) : ('load' as const),
    storeOp: 'store' as const,
  };
}

function draw() {
  fillPipeline
    .with(bindGroup)
    .withColorAttachment(colorAttachment(true))
    .withDepthStencilAttachment({
      view: depthTextureView,
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    })
    .drawIndexed(roundedBoxIndexCountPerPatch(subdivLevel), instanceCount);

  wireframePipeline
    .with(bindGroup)
    .withColorAttachment(colorAttachment(false))
    .withDepthStencilAttachment({
      view: depthTextureView,
      depthLoadOp: 'load',
      depthStoreOp: 'store',
    })
    .drawIndexed(roundedBoxWireframeIndexCountPerPatch(subdivLevel), instanceCount);
}

let destroyed = false;
function frame() {
  if (destroyed) {
    return;
  }
  draw();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// #region Example controls & Cleanup

export const controls = defineControls({
  Subdivisions: {
    initial: subdivLevel,
    min: 1,
    max: MAX_SUBDIV,
    step: 1,
    onSliderChange: (newValue) => {
      subdivLevel = newValue;
      uniforms.write({ subdivCount: subdivLevel });
    },
  },
  'Border radius': {
    initial: cornerRadius,
    min: 0,
    max: 2.5,
    step: 0.05,
    onSliderChange: (newValue) => {
      cornerRadius = newValue;
      boxes.write([
        RoundedBox({
          position: d.vec3f(0, 0, 0),
          halfSize,
          cornerRadius,
        }),
      ]);
    },
  },
});

export function onCleanup() {
  destroyed = true;
  resizeObserver.disconnect();
  cleanupCamera();
  root.destroy();
}

// #endregion
