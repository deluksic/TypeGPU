import {
  icosphereIndexCountPerFace,
  icosphereInstanceCount,
  icosphere,
  icosphereWireframeIndexCountPerFace,
  subdivSphericalTriangleSlot,
  subdivSphericalTriangles,
} from '@typegpu/geometry';
import { subdivTriangleIndices, subdivTriangleWireframeIndices } from '@typegpu/geometry';
import tgpu, { d, std as s } from 'typegpu';
import { Camera, setupOrbitCamera } from '../../common/setup-orbit-camera.ts';
import { defineControls } from '../../common/defineControls.ts';

const root = await tgpu.init();
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const multisample = false;

const MAX_ICOSPHERE_SUBDIV = 10;

const Uniforms = d.struct({
  subdivCount: d.u32,
});

const Sphere = d.struct({
  position: d.vec3f,
  radius: d.f32,
});

const bindGroupLayout = tgpu.bindGroupLayout({
  uniforms: {
    uniform: Uniforms,
  },
  camera: {
    uniform: Camera,
  },
  spheres: {
    storage: (n: number) => d.arrayOf(Sphere, n),
  },
});

const indexBuffer = root
  .createBuffer(
    d.arrayOf(d.u32, subdivTriangleIndices(MAX_ICOSPHERE_SUBDIV).length),
    subdivTriangleIndices(MAX_ICOSPHERE_SUBDIV),
  )
  .$usage('index');

const wireframeIndexBuffer = root
  .createBuffer(
    d.arrayOf(d.u32, subdivTriangleWireframeIndices(MAX_ICOSPHERE_SUBDIV).length),
    subdivTriangleWireframeIndices(MAX_ICOSPHERE_SUBDIV),
  )
  .$usage('index');

const sphereCount = 10;

const spheres = root
  .createBuffer(
    d.arrayOf(Sphere, sphereCount),
    Array.from({ length: sphereCount }).map(() =>
      Sphere({
        position: d.vec3f(
          (Math.random() - 0.5) * 24,
          (Math.random() - 0.5) * 24,
          (Math.random() - 0.5) * 24,
        ),
        radius: 2 + Math.random() * 0.9,
      }),
    ),
  )
  .$usage('storage');

let subdivLevel = 4;
let lift = subdivSphericalTriangles.uniformArea;

const uniforms = root.createBuffer(Uniforms, { subdivCount: subdivLevel }).$usage('uniform');
const camera = root.createBuffer(Camera).$usage('uniform');

const bindGroup = root.createBindGroup(bindGroupLayout, {
  uniforms,
  camera,
  spheres,
});

const { cleanupCamera } = setupOrbitCamera(
  canvas,
  {
    initPos: d.vec4f(0, 8, 24, 1),
    target: d.vec4f(0, 0, 0, 1),
    minZoom: 8,
    maxZoom: 80,
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
    sphereIndex: d.interpolate('flat', d.u32),
  },
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const subdivCount = bindGroupLayout.$.uniforms.subdivCount;
  const ico = icosphere(instanceIndex, vertexIndex, subdivCount);
  const sphere = bindGroupLayout.$.spheres[ico.instanceIndex];
  const worldPos = sphere.position + ico.vertex * sphere.radius;
  const cameraUniform = bindGroupLayout.$.camera;
  return {
    outPos: cameraUniform.projection * cameraUniform.view * d.vec4f(worldPos, 1),
    worldNormal: ico.vertex,
    sphereIndex: ico.instanceIndex,
  };
});

const mainFragment = tgpu.fragmentFn({
  in: {
    worldNormal: d.vec3f,
    sphereIndex: d.interpolate('flat', d.u32),
  },
  out: d.vec4f,
})(({ worldNormal, sphereIndex }) => {
  'use gpu';
  const lightDir = s.normalize(d.vec3f(0.4, 1, 0.3));
  const diffuse = s.max(0, s.dot(worldNormal, lightDir));
  const ambient = 0.35;
  const color = d.vec3f(1, s.cos(d.f32(sphereIndex)), s.sin(5 * d.f32(sphereIndex)));
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

function createPipelines() {
  const pipelineRoot = root.with(subdivSphericalTriangleSlot, lift);

  const fillPipeline = pipelineRoot
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

  const wireframePipeline = pipelineRoot
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

  return { fillPipeline, wireframePipeline };
}

let { fillPipeline, wireframePipeline } = createPipelines();

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

const instanceCount = icosphereInstanceCount(sphereCount);

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
    .drawIndexed(icosphereIndexCountPerFace(subdivLevel), instanceCount);

  wireframePipeline
    .with(bindGroup)
    .withColorAttachment(colorAttachment(false))
    .withDepthStencilAttachment({
      view: depthTextureView,
      depthLoadOp: 'load',
      depthStoreOp: 'store',
    })
    .drawIndexed(icosphereWireframeIndexCountPerFace(subdivLevel), instanceCount);
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
    max: MAX_ICOSPHERE_SUBDIV,
    step: 1,
    onSliderChange: (newValue) => {
      subdivLevel = newValue;
      uniforms.write({ subdivCount: subdivLevel });
    },
  },
  Lift: {
    initial: 'uniformArea',
    options: Object.keys(subdivSphericalTriangles),
    onSelectChange: (selected) => {
      lift = subdivSphericalTriangles[selected as keyof typeof subdivSphericalTriangles];
      ({ fillPipeline, wireframePipeline } = createPipelines());
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
