import {
  capsule,
  CAPSULE_CAP_COUNT,
  CAPSULE_EDGE_PATCH_OFFSET,
  CAPSULE_PATCH_COUNT,
  capsuleInstanceCount,
  capsuleIndexCountPerPatch,
  capsuleWireframeIndexCountPerPatch,
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

const Capsule = d.struct({
  position: d.vec3f,
  radius: d.f32,
  cylHalf: d.f32,
});

const bindGroupLayout = tgpu.bindGroupLayout({
  uniforms: {
    uniform: Uniforms,
  },
  camera: {
    uniform: Camera,
  },
  capsules: {
    storage: (n: number) => d.arrayOf(Capsule, n),
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

const capsuleCount = 1;

let radius = 1.2;
let cylHalf = 2.5;

const capsules = root
  .createBuffer(
    d.arrayOf(Capsule, capsuleCount),
    [
      Capsule({
        position: d.vec3f(0, 0, 0),
        radius,
        cylHalf,
      }),
    ],
  )
  .$usage('storage');

let subdivLevel = 4;

const instanceCount = capsuleInstanceCount(capsuleCount);

const uniforms = root.createBuffer(Uniforms, { subdivCount: subdivLevel }).$usage('uniform');
const camera = root.createBuffer(Camera).$usage('uniform');

const bindGroup = root.createBindGroup(bindGroupLayout, {
  uniforms,
  camera,
  capsules,
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
  const objectIndex = d.u32(instanceIndex / CAPSULE_PATCH_COUNT);
  const capsuleData = bindGroupLayout.$.capsules[objectIndex];
  const patchIndex = d.u32(instanceIndex % CAPSULE_PATCH_COUNT);
  const patch = capsule(
    instanceIndex,
    vertexIndex,
    subdivCount,
    capsuleData.radius,
    capsuleData.cylHalf,
  );
  const worldPos = capsuleData.position + patch.vertex;
  const cameraUniform = bindGroupLayout.$.camera;
  return {
    outPos: cameraUniform.projection * cameraUniform.view * d.vec4f(worldPos, 1),
    worldNormal: patch.normal,
    patchIndex,
  };
});

/** Patch color key: caps 0–7 (top then bottom octants), edges 8–15 (four side quads × 2 tris). */
const patchBaseColor = tgpu.fn([d.u32], d.vec3f)((patchIndex) => {
  'use gpu';
  if (patchIndex < CAPSULE_CAP_COUNT) {
    const hue = d.f32(patchIndex) * d.f32(0.785398);
    return d.vec3f(
      0.55 + 0.25 * s.cos(hue),
      0.55 + 0.25 * s.cos(hue + d.f32(2.094)),
      0.55 + 0.25 * s.cos(hue + d.f32(4.189)),
    );
  }
  const edgeIndex = (patchIndex - CAPSULE_EDGE_PATCH_OFFSET) >> d.u32(1);
  const hue = d.f32(edgeIndex) * d.f32(1.570796);
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
    .drawIndexed(capsuleIndexCountPerPatch(subdivLevel), instanceCount);

  wireframePipeline
    .with(bindGroup)
    .withColorAttachment(colorAttachment(false))
    .withDepthStencilAttachment({
      view: depthTextureView,
      depthLoadOp: 'load',
      depthStoreOp: 'store',
    })
    .drawIndexed(capsuleWireframeIndexCountPerPatch(subdivLevel), instanceCount);
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

function writeCapsule() {
  capsules.write([
    Capsule({
      position: d.vec3f(0, 0, 0),
      radius,
      cylHalf,
    }),
  ]);
}

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
  Radius: {
    initial: radius,
    min: 0.2,
    max: 3,
    step: 0.05,
    onSliderChange: (newValue) => {
      radius = newValue;
      writeCapsule();
    },
  },
  'Cylinder half-length': {
    initial: cylHalf,
    min: 0,
    max: 5,
    step: 0.05,
    onSliderChange: (newValue) => {
      cylHalf = newValue;
      writeCapsule();
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
