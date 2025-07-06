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
  mul,
  normalize,
  select,
  sign,
  sin,
  sub,
} from 'typegpu/std';
import {
  addMul,
  cross2d,
  externalNormals,
  intersectLines,
  limitAlong,
  miterPoint,
  ortho2d,
} from './utils.ts';

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
    position: d.vec2f(-0.9, 0),
    radius: 0.05,
  }),
  LineVertex({
    position: d.vec2f(-0.8, 0),
    radius: 0.3,
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
    position: d.vec2f(0.8, 0),
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
  0, 4, 1,
  4, 7, 1,
  6, 7, 4,
  1, 7, 2,
  2, 7, 8,
  3, 2, 5,
  5, 2, 8,
  8, 9, 5,
];
const indexBuffer = root.createBuffer(d.arrayOf(d.u16, indices.length), indices)
  .$usage('index');

// deno-fmt-ignore
const outlineIndices = [
  0, 1,
  0, 4,
  1, 4,
  4, 7,
  4, 6,
  6, 7,
  1, 2,
  1, 7,
  2, 7,
  2, 8,
  7, 8,
  2, 3,
  2, 5,
  3, 5,
  5, 8,
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
  // if (instanceIndex === 0) {
  //   return {
  //     outPos: d.vec4f(0, 0, 0, 0),
  //     instanceIndex: 0,
  //   };
  // }
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

  const externalAB = externalNormals(AB, A.radius, B.radius);
  const externalBC = externalNormals(BC, B.radius, C.radius);
  const externalCD = externalNormals(CD, C.radius, D.radius);

  const nAB1 = externalAB.n1;
  const nAB2 = externalAB.n2;
  const nBC1 = externalBC.n1;
  const nBC2 = externalBC.n2;
  const nCD1 = externalCD.n1;
  const nCD2 = externalCD.n2;

  const tBC1 = ortho2d(nBC1);
  const tBC2 = ortho2d(nBC2);

  const mB1 = miterPoint(nAB1, nBC1);
  const mB2 = miterPoint(nAB2, nBC2);
  const mC1 = miterPoint(nBC1, nCD1);
  const mC2 = miterPoint(nBC2, nCD2);

  const nmB1 = normalize(mB1);
  const nmB2 = normalize(mB2);
  const nmC1 = normalize(mC1);
  const nmC2 = normalize(mC2);

  const cmB1 = cross2d(nAB1, nBC1);
  const cmB2 = cross2d(nAB2, nBC2);
  const cmC1 = cross2d(nBC1, nCD1);
  const cmC2 = cross2d(nBC2, nCD2);

  let hB1 = nmB1;
  let hB2 = nmB2;
  let hC1 = nmC1;
  let hC2 = nmC2;

  const flip = sign(B.radius - C.radius);
  const cAB2BC1 = flip * cross2d(nAB2, nBC1) > 0;
  const cAB1BC2 = flip * cross2d(nAB1, nBC2) < 0;
  const h_ = normalize(BC);
  const h = mul(h_, -flip);
  if (dot(nmB1, nmB2) > 0) {
    if (cAB2BC1) {
      if (cAB1BC2) {
        hB1 = h;
      } else {
        hB1 = mul(hB1, -1);
      }
    } else {
      if (!cAB1BC2) {
        hB1 = h_;
      }
    }
    if (cAB1BC2) {
      if (cAB2BC1) {
        hB2 = h;
      } else {
        hB2 = mul(hB2, -1);
      }
    } else {
      if (!cAB2BC1) {
        hB2 = h_;
      }
    }
  }

  const cBC1CD2 = flip * cross2d(nBC1, nCD2) < 0;
  const cBC2CD1 = flip * cross2d(nBC2, nCD1) > 0;
  if (dot(nmC1, nmC2) > 0) {
    if (cBC1CD2) {
      if (cBC2CD1) {
        hC1 = h;
      } else {
        hC1 = mul(hC1, -1);
      }
    } else {
      if (!cBC2CD1) {
        hC1 = h_;
      }
    }
    if (cBC2CD1) {
      if (cBC1CD2) {
        hC2 = h;
      } else {
        hC2 = mul(hC2, -1);
      }
    } else {
      if (!cBC1CD2) {
        hC2 = h_;
      }
    }
  }

  const reverseMiterB1 = dot(nmB1, nmB2) < 0 && cmB1 > 0;
  const reverseMiterB2 = dot(nmB1, nmB2) < 0 && cmB2 < 0;
  const reverseMiterC1 = dot(nmC1, nmC2) < 0 && cmC1 > 0;
  const reverseMiterC2 = dot(nmC1, nmC2) < 0 && cmC2 < 0;

  const v0 = addMul(B.position, select(hB1, mB1, reverseMiterB1), B.radius);
  const v1 = addMul(B.position, select(nBC1, mB1, reverseMiterB1), B.radius);
  const v2 = addMul(C.position, select(nBC1, mC1, reverseMiterC1), C.radius);
  const v3 = addMul(C.position, select(hC1, mC1, reverseMiterC1), C.radius);

  const v6 = addMul(B.position, select(hB2, mB2, reverseMiterB2), B.radius);
  const v7 = addMul(B.position, select(nBC2, mB2, reverseMiterB2), B.radius);
  const v8 = addMul(C.position, select(nBC2, mC2, reverseMiterC2), C.radius);
  const v9 = addMul(C.position, select(hC2, mC2, reverseMiterC2), C.radius);

  const centerB = intersectLines(nAB1, nAB2, nBC1, nBC2);
  const centerC = intersectLines(nCD1, nCD2, nBC1, nBC2);
  const v4 = select(
    B.position,
    addMul(B.position, centerB.point, B.radius),
    centerB.valid && centerB.t >= 0 && centerB.t <= 1,
  );
  const v5 = select(
    C.position,
    addMul(C.position, centerC.point, C.radius),
    centerC.valid && centerC.t >= 0 && centerC.t <= 1,
  );

  const tC1limit = addMul(C.position, nBC1, C.radius);
  const tC2limit = addMul(C.position, nBC2, C.radius);
  const v0fix = limitAlong(v0, tC1limit, tBC1, false);
  const v1fix = limitAlong(v1, tC1limit, tBC1, false);
  const v6fix = limitAlong(v6, tC2limit, tBC2, true);
  const v7fix = limitAlong(v7, tC2limit, tBC2, true);

  const tB1limit = addMul(B.position, nBC1, B.radius);
  const tB2limit = addMul(B.position, nBC2, B.radius);
  const v2fix = limitAlong(v2, tB1limit, tBC1, true);
  const v3fix = limitAlong(v3, tB1limit, tBC1, true);
  const v8fix = limitAlong(v8, tB2limit, tBC2, false);
  const v9fix = limitAlong(v9, tB2limit, tBC2, false);

  const v4fix = select(v4, v1, reverseMiterB1 && reverseMiterB2);
  const v4fixfix = select(
    v4fix,
    addMul(B.position, h_, B.radius),
    dot(nmB1, nmB2) > 0 && !cAB1BC2 && !cAB2BC1,
  );

  const v5fix = select(v5, v2, reverseMiterC1 && reverseMiterC2);
  const v5fixfix = select(
    v5fix,
    addMul(C.position, h_, C.radius),
    dot(nmC1, nmC2) > 0 && !cBC1CD2 && !cBC2CD1,
  );

  const points = [
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
  ];

  const pointsFix = [
    v0fix,
    v1fix,
    v2fix,
    v3fix,
    v4fixfix,
    v5fixfix,
    v6fix,
    v7fix,
    v8fix,
    v9fix,
  ];

  return {
    outPos: d.vec4f(pointsFix[vertexIndex], 0.0, 1.0),
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
    d.vec3f(1, 0, 0),
    d.vec3f(0, 1, 0),
    d.vec3f(0, 0, 1),
    d.vec3f(1, 0, 1),
  ];
  const color = colors[(instanceIndex % 12) % 4];
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
    cullMode: 'back',
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

const draw = (frameId: number) => {
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
    .drawIndexed(32, 4);

  circlesPipeline
    .with(bindGroupLayout, uniformsBindGroup)
    .withColorAttachment(colorAttachment)
    .draw(CIRCLE_SEGMENT_COUNT + 1, 4 + 1);
};

let frameId = 0;
const runAnimationFrame = () => {
  draw(frameId++);
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
