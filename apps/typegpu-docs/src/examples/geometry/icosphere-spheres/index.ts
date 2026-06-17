import {
  cubesphereInstanceCount,
  icosphereInstanceCount,
  octasphereInstanceCount,
  proceduralSphereObjectIndices,
  proceduralSpheres,
  sphereObjectIndexSlot,
  sphereSlot,
  segmentSphericalTriangleSlot,
  segmentSphericalTriangles,
  segmentTriangleIndices,
  segmentTriangleIndexCount,
  segmentTriangleWireframeIndices,
  segmentTriangleWireframeIndexCount,
} from '@typegpu/geometry';
import tgpu, { d, std as s } from 'typegpu';
import { Camera, setupOrbitCamera } from '../../common/setup-orbit-camera.ts';
import { defineControls } from '../../common/defineControls.ts';

const root = await tgpu.init();
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const multisample = true;

const MAX_SPHERE_SEGMENT_COUNT = 16;

const Uniforms = d.struct({
  segmentCount: d.u32,
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
    d.arrayOf(d.u32, segmentTriangleIndices(MAX_SPHERE_SEGMENT_COUNT).length),
    segmentTriangleIndices(MAX_SPHERE_SEGMENT_COUNT),
  )
  .$usage('index');

const wireframeIndexBuffer = root
  .createBuffer(
    d.arrayOf(d.u32, segmentTriangleWireframeIndices(MAX_SPHERE_SEGMENT_COUNT).length),
    segmentTriangleWireframeIndices(MAX_SPHERE_SEGMENT_COUNT),
  )
  .$usage('index');

const sphereCount = 30;

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
        radius: 1 + Math.random() * 0.9,
      }),
    ),
  )
  .$usage('storage');

let segmentCount = 4;
let wireframe = true;
let lift = segmentSphericalTriangles.uniformArea;
let sphereKind: keyof typeof proceduralSpheres = 'icosphere';
let sphere = proceduralSpheres[sphereKind];
let sphereObjectIndex = proceduralSphereObjectIndices[sphereKind];

type SphereKind = keyof typeof proceduralSpheres;

function sphereDrawLayout(kind: SphereKind) {
  const instanceCount =
    kind === 'icosphere'
      ? icosphereInstanceCount(sphereCount)
      : kind === 'octasphere'
        ? octasphereInstanceCount(sphereCount)
        : cubesphereInstanceCount(sphereCount);

  return {
    indexBuffer,
    wireframeIndexBuffer,
    indexCountPerFace: segmentTriangleIndexCount,
    wireframeIndexCountPerFace: segmentTriangleWireframeIndexCount,
    instanceCount,
  };
}

let drawLayout = sphereDrawLayout(sphereKind);

const uniforms = root.createBuffer(Uniforms, { segmentCount: segmentCount }).$usage('uniform');
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
  const segmentCount = bindGroupLayout.$.uniforms.segmentCount;
  const objectIndex = sphereObjectIndexSlot.$(instanceIndex);
  const patch = sphereSlot.$(instanceIndex, vertexIndex, segmentCount);
  const sphereData = bindGroupLayout.$.spheres[objectIndex];
  const worldPos = sphereData.position + patch.vertex * sphereData.radius;
  const cameraUniform = bindGroupLayout.$.camera;
  return {
    outPos: cameraUniform.projection * cameraUniform.view * d.vec4f(worldPos, 1),
    worldNormal: patch.vertex,
    sphereIndex: objectIndex,
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
  let pipelineRoot = root.with(sphereSlot, sphere).with(sphereObjectIndexSlot, sphereObjectIndex);
  if (sphereKind === 'icosphere' || sphereKind === 'octasphere') {
    pipelineRoot = pipelineRoot.with(segmentSphericalTriangleSlot, lift);
  }

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
    .withIndexBuffer(drawLayout.indexBuffer);

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
    .withIndexBuffer(drawLayout.wireframeIndexBuffer);

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
    .drawIndexed(drawLayout.indexCountPerFace(segmentCount), drawLayout.instanceCount);

  if (wireframe) {
    wireframePipeline
      .with(bindGroup)
      .withColorAttachment(colorAttachment(false))
      .withDepthStencilAttachment({
        view: depthTextureView,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      })
      .drawIndexed(drawLayout.wireframeIndexCountPerFace(segmentCount), drawLayout.instanceCount);
  }
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
  Sphere: {
    initial: sphereKind,
    options: Object.keys(proceduralSpheres),
    onSelectChange: (selected) => {
      sphereKind = selected as SphereKind;
      sphere = proceduralSpheres[sphereKind];
      sphereObjectIndex = proceduralSphereObjectIndices[sphereKind];
      drawLayout = sphereDrawLayout(sphereKind);
      ({ fillPipeline, wireframePipeline } = createPipelines());
    },
  },
  Segments: {
    initial: segmentCount,
    min: 1,
    max: MAX_SPHERE_SEGMENT_COUNT,
    step: 1,
    onSliderChange: (newValue) => {
      segmentCount = newValue;
      uniforms.write({ segmentCount: segmentCount });
    },
  },
  Lift: {
    initial: 'uniformArea',
    options: Object.keys(segmentSphericalTriangles),
    onSelectChange: (selected) => {
      lift = segmentSphericalTriangles[selected as keyof typeof segmentSphericalTriangles];
      ({ fillPipeline, wireframePipeline } = createPipelines());
    },
  },
  Wireframe: {
    initial: wireframe,
    onToggleChange: (value) => {
      wireframe = value;
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
