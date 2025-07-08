import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import type { ColorAttachment } from '../../../../../../../packages/typegpu/src/core/pipeline/renderPipeline.ts';
import {
  arrayLength,
  clamp,
  cos,
  dot,
  max,
  min,
  normalize,
  select,
  sin,
  sub,
} from 'typegpu/std';
import {
  addMul,
  externalNormals,
  limitAlong,
  ortho2d,
  ortho2dNeg,
} from './utils.ts';
import { solveCap, solveJoin } from './lines.ts';

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
console.log(`Using ${adapter?.info.vendor} adapter`);
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

// Textures
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

const LineVertex = d.struct({
  position: d.vec2f,
  radius: d.f32,
});

const bindGroupLayout = tgpu.bindGroupLayout({
  lineVertices: {
    storage: (n: number) => d.arrayOf(LineVertex, n),
  },
});

const lineVertices = [
  LineVertex({
    position: d.vec2f(-0.7, 0),
    radius: 0.05,
  }),
  LineVertex({
    position: d.vec2f(-0.6, 0),
    radius: 0.25,
  }),
  LineVertex({
    position: d.vec2f(-0.2, 0),
    radius: 0.1,
  }),
  LineVertex({
    position: d.vec2f(0.6, 0),
    radius: 0.2,
  }),
  LineVertex({
    position: d.vec2f(0.4, 0.4),
    radius: 0.05,
  }),
];
const lineVerticesBuffer = root.createBuffer(
  d.arrayOf(LineVertex, lineVertices.length),
  lineVertices,
).$usage('storage');

const uniformsBindGroup = root.createBindGroup(bindGroupLayout, {
  lineVertices: lineVerticesBuffer,
});

// deno-fmt-ignore
const indices = [
  0, 2, 1,
  1, 2, 3,
  2, 4, 3,
  3, 6, 1,
  6, 3, 8,
  5, 6, 7,
  7, 6, 8,
  8, 9, 7,
];
const indexBuffer = root.createBuffer(d.arrayOf(d.u16, indices.length), indices)
  .$usage('index');

// deno-fmt-ignore
const outlineIndices = [
  0, 1,
  0, 2,
  1, 2,
  1, 3,
  2, 3,
  2, 4,
  3, 4,
  1, 6,
  3, 6,
  3, 8,
  6, 8,
  5, 6,
  5, 7,
  6, 7,
  7, 8,
  7, 9,
  8, 9,
];
const outlineIndexBuffer = root.createBuffer(
  d.arrayOf(d.u16, outlineIndices.length),
  outlineIndices,
).$usage('index');

const mainVertex = tgpu['~unstable'].vertexFn({
  in: {
    instanceIndex: d.builtin.instanceIndex,
    vertexIndex: d.builtin.vertexIndex,
  },
  out: {
    outPos: d.builtin.position,
    instanceIndex: d.interpolate('flat', d.u32),
  },
})(({ vertexIndex, instanceIndex }) => {
  const firstIndex = max(0, d.i32(instanceIndex) - 1);
  const lastIndex = min(
    arrayLength(bindGroupLayout.$.lineVertices) - 1,
    instanceIndex + 2,
  );
  const A = bindGroupLayout.$.lineVertices[firstIndex];
  const B = bindGroupLayout.$.lineVertices[instanceIndex];
  const C = bindGroupLayout.$.lineVertices[instanceIndex + 1];
  const D = bindGroupLayout.$.lineVertices[lastIndex];

  const AB = sub(B.position, A.position);
  const BC = sub(C.position, B.position);
  const CD = sub(D.position, C.position);

  const radiusABDelta = A.radius - B.radius;
  const radiusBCDelta = B.radius - C.radius;
  const radiusCDDelta = C.radius - D.radius;

  // segment where one end completely contains the other
  // is skipped
  if (dot(BC, BC) < radiusBCDelta * radiusBCDelta) {
    return {
      outPos: d.vec4f(0, 0, 0, 0),
      instanceIndex: 0,
    };
  }

  const isCapB = dot(AB, AB) <= radiusABDelta * radiusABDelta;
  const isCapC = dot(CD, CD) <= radiusCDDelta * radiusCDDelta;

  const eAB = externalNormals(AB, A.radius, B.radius);
  const eBC = externalNormals(BC, B.radius, C.radius);
  const eCD = externalNormals(CD, C.radius, D.radius);

  const nAB = normalize(AB);
  const nBC = normalize(BC);

  let joinB = solveCap(eBC.n1, eBC.n2);
  if (!isCapB) {
    joinB = solveJoin(nAB, eAB.n1, eBC.n1, eAB.n2, eBC.n2);
  }

  let v0 = addMul(B.position, joinB.u, B.radius);
  let v1 = addMul(B.position, joinB.uR, B.radius);
  const v2 = addMul(B.position, joinB.c, B.radius);
  let v3 = addMul(B.position, joinB.dR, B.radius);
  let v4 = addMul(B.position, joinB.d, B.radius);

  let joinC = solveCap(eBC.n2, eBC.n1);
  if (!isCapC) {
    joinC = solveJoin(nBC, eBC.n1, eCD.n1, eBC.n2, eCD.n2);
  }

  let v5 = addMul(C.position, joinC.u, C.radius);
  let v6 = addMul(C.position, joinC.uL, C.radius);
  const v7 = addMul(C.position, joinC.c, C.radius);
  let v8 = addMul(C.position, joinC.dL, C.radius);
  let v9 = addMul(C.position, joinC.d, C.radius);

  const tBC1 = ortho2d(eBC.n1);
  const tBC2 = ortho2dNeg(eBC.n2);

  if (!isCapB) {
    const limB1 = addMul(C.position, eBC.n1, C.radius);
    const limB2 = addMul(C.position, eBC.n2, C.radius);
    v0 = limitAlong(v0, limB1, tBC1, false);
    v1 = limitAlong(v1, limB1, tBC1, false);
    v3 = limitAlong(v3, limB2, tBC2, false);
    v4 = limitAlong(v4, limB2, tBC2, false);
  }

  if (!isCapC) {
    const limC1 = addMul(B.position, eBC.n1, B.radius);
    const limC2 = addMul(B.position, eBC.n2, B.radius);
    v5 = limitAlong(v5, limC1, tBC1, true);
    v6 = limitAlong(v6, limC1, tBC1, true);
    v8 = limitAlong(v8, limC2, tBC2, true);
    v9 = limitAlong(v9, limC2, tBC2, true);
  }

  const points = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9];

  return {
    outPos: d.vec4f(points[vertexIndex], 0.0, 1.0),
    instanceIndex,
  };
});

console.log(tgpu.resolve({ externals: { mainVertex } }));

const mainFragment = tgpu['~unstable'].fragmentFn({
  in: {
    instanceIndex: d.interpolate('flat', d.u32),
    frontFacing: d.builtin.frontFacing,
    position: d.builtin.position,
  },
  out: d.vec4f,
})(({ position, instanceIndex, frontFacing }) => {
  const colors = [
    d.vec3f(1, 0, 0), // 0
    d.vec3f(0, 1, 0), // 1
    d.vec3f(0, 0, 1), // 2
    d.vec3f(1, 0, 1), // 3
    d.vec3f(1, 1, 0), // 4
    d.vec3f(0, 1, 1), // 5
    d.vec3f(0, 0, 0),
  ];
  const color = colors[(instanceIndex % 12) % 7];
  if (frontFacing) {
    return d.vec4f(color, 0.2);
  }
  return d.vec4f(
    color,
    select(
      d.f32(0),
      d.f32(0.2),
      ((d.u32(position.x) >> 3) % 2) !== ((d.u32(position.y) >> 3) % 2),
    ),
  );
});

const outlineFragment = tgpu['~unstable'].fragmentFn({
  in: {
    _unused: d.builtin.frontFacing,
  },
  out: d.vec4f,
})(() => {
  return d.vec4f(0, 0, 0, 1);
});

const alphaBlend: GPUBlendState = {
  color: {
    operation: 'add',
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
  },
  alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
};

const pipeline = root['~unstable']
  .withVertex(mainVertex, {})
  .withFragment(mainFragment, {
    format: presentationFormat,
    blend: alphaBlend,
  })
  .withMultisample({ count: multisample ? 4 : 1 })
  .withPrimitive({
    // cullMode: 'back',
  })
  .createPipeline()
  .withIndexBuffer(indexBuffer);

const outlinePipeline = root['~unstable']
  .withVertex(mainVertex, {})
  .withFragment(outlineFragment, {
    format: presentationFormat,
    blend: alphaBlend,
  })
  .withMultisample({ count: multisample ? 4 : 1 })
  .withPrimitive({
    topology: 'line-list',
  })
  .createPipeline()
  .withIndexBuffer(outlineIndexBuffer);

const CIRCLE_SEGMENT_COUNT = 256;
const CIRCLE_MIN_STEP = 2 * Math.PI / CIRCLE_SEGMENT_COUNT;
const CIRCLE_MAX_STEP = Math.PI / 8;
const CIRCLE_DASH_LEN = 0.0025 * Math.PI;

const circlesVertex = tgpu['~unstable'].vertexFn({
  in: {
    instanceIndex: d.builtin.instanceIndex,
    vertexIndex: d.builtin.vertexIndex,
  },
  out: {
    outPos: d.builtin.position,
  },
})(({ instanceIndex, vertexIndex }) => {
  const vertex = bindGroupLayout.$.lineVertices[instanceIndex];
  const step = clamp(
    CIRCLE_DASH_LEN / vertex.radius,
    CIRCLE_MIN_STEP,
    CIRCLE_MAX_STEP,
  );
  const angle = min(2 * Math.PI, step * d.f32(vertexIndex));
  const unit = d.vec2f(cos(angle), sin(angle));
  return {
    outPos: d.vec4f(addMul(vertex.position, unit, vertex.radius), 0, 1),
  };
});

const circlesPipeline = root['~unstable']
  .withVertex(circlesVertex, {})
  .withFragment(outlineFragment, {
    format: presentationFormat,
    blend: alphaBlend,
  })
  .withMultisample({ count: multisample ? 4 : 1 })
  .withPrimitive({
    topology: 'line-list',
  })
  .createPipeline();

const draw = () => {
  const colorAttachment: ColorAttachment = {
    ...(multisample
      ? {
        view: msaaTextureView,
        resolveTarget: context.getCurrentTexture().createView(),
      }
      : {
        view: context.getCurrentTexture().createView(),
      }),
    clearValue: [0, 0, 0, 0],
    loadOp: 'load',
    storeOp: 'store',
  };
  pipeline
    .with(bindGroupLayout, uniformsBindGroup)
    .withColorAttachment({ ...colorAttachment, loadOp: 'clear' })
    .drawIndexed(24, 4);

  outlinePipeline
    .with(bindGroupLayout, uniformsBindGroup)
    .withColorAttachment(colorAttachment)
    .drawIndexed(outlineIndices.length, 4);

  circlesPipeline
    .with(bindGroupLayout, uniformsBindGroup)
    .withColorAttachment(colorAttachment)
    .draw(CIRCLE_SEGMENT_COUNT + 1, 4 + 1);
};

const runAnimationFrame = () => {
  draw();
  requestAnimationFrame(runAnimationFrame);
};
runAnimationFrame();

canvas.addEventListener('pointermove', (ev) => {
  const rect = canvas.getBoundingClientRect();
  const mx = 2 * (ev.clientX - rect.left) / rect.width - 1;
  const my = 2 * (rect.bottom - ev.clientY) / rect.height - 1;
  lineVerticesBuffer.writePartial([{
    idx: 1,
    value: { position: d.vec2f(mx, my) },
  }]);
});

export function onCleanup() {
  root.destroy();
}
