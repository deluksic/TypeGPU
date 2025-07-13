import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import type { ColorAttachment } from '../../../../../../../packages/typegpu/src/core/pipeline/renderPipeline.ts';
import { arrayLength, clamp, cos, max, min, select, sin } from 'typegpu/std';
import {
  lineSegmentIndicesCapLevel1,
  lineSegmentIndicesCapLevel2,
  lineSegmentVariableWidth,
  LineSegmentVertex,
  lineSegmentWireframeIndicesCapLevel2,
  lineSingleSegmentVariableWidth,
} from '@typegpu/geometry';
import { addMul } from '../../../../../../../packages/typegpu-geometry/src/utils.ts';

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

const Uniforms = d.struct({
  time: d.f32,
});

const uniformsBuffer = root.createBuffer(Uniforms, {
  time: 0,
}).$usage('uniform');

const bindGroupLayout = tgpu.bindGroupLayout({
  lineVertices: {
    storage: (n: number) => d.arrayOf(LineSegmentVertex, n),
  },
  uniforms: {
    uniform: Uniforms,
  },
});

const lineVertices = [
  LineSegmentVertex({
    position: d.vec2f(-0.6, 0),
    radius: 0.1,
  }),
  LineSegmentVertex({
    position: d.vec2f(-0.4, 0),
    radius: 0.25,
  }),
  LineSegmentVertex({
    position: d.vec2f(-0.2, 0),
    radius: 0.1,
  }),
  LineSegmentVertex({
    position: d.vec2f(0.6, -0.2),
    radius: 0.1,
  }),
  LineSegmentVertex({
    position: d.vec2f(0.2, 0.3),
    radius: 0.1,
  }),
];
const lineVerticesBuffer = root.createBuffer(
  d.arrayOf(LineSegmentVertex, lineVertices.length),
  lineVertices,
).$usage('storage');

const uniformsBindGroup = root.createBindGroup(bindGroupLayout, {
  lineVertices: lineVerticesBuffer,
  uniforms: uniformsBuffer,
});

const indexBuffer = root.createBuffer(
  d.arrayOf(d.u16, lineSegmentIndicesCapLevel2.length),
  lineSegmentIndicesCapLevel2,
)
  .$usage('index');

const outlineIndexBuffer = root.createBuffer(
  d.arrayOf(d.u16, lineSegmentWireframeIndicesCapLevel2.length),
  lineSegmentWireframeIndicesCapLevel2,
).$usage('index');

const animation = tgpu.fn([d.f32, d.f32, d.f32, d.f32], LineSegmentVertex)(
  (i, t, fx, fy) => {
    return LineSegmentVertex({
      position: d.vec2f(0.8 * cos(0.1 * fx * i), 0.8 * sin(0.1 * fy * i)),
      radius: 0.05 * clamp(sin(6 * Math.PI * i / 200 + Math.PI * t), 0, 0.5) +
        0.01,
    });
  },
);

// const animation = tgpu.fn([d.f32, d.f32, d.f32, d.f32], LineSegmentVertex)(
//   (i, t, fx, fy) => {
//     return LineSegmentVertex({
//       position: d.vec2f(fx * cos(0.2 * fy * i), fx * sin(0.2 * fy * i)),
//       radius: 0.05 * clamp(sin(0.1 * i), 0, 0.5) + 0.01,
//     });
//   },
// );

const mainVertex = tgpu['~unstable'].vertexFn({
  in: {
    instanceIndex: d.builtin.instanceIndex,
    vertexIndex: d.builtin.vertexIndex,
  },
  out: {
    outPos: d.builtin.position,
    instanceIndex: d.interpolate('flat', d.u32),
    uv: d.vec2f,
  },
})(({ vertexIndex, instanceIndex }) => {
  const t = bindGroupLayout.$.uniforms.time;
  // if (instanceIndex !== 2) {
  //   return {
  //     outPos: d.vec4f(),
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
  // const v = bindGroupLayout.$.lineVertices[0].position;
  // const A = animation(d.f32(instanceIndex), t, v.x, v.y);
  // const B = animation(d.f32(instanceIndex + 1), t, v.x, v.y);
  // const C = animation(d.f32(instanceIndex + 2), t, v.x, v.y);
  // const D = animation(d.f32(instanceIndex + 3), t, v.x, v.y);

  const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D);

  return {
    outPos: d.vec4f(result.vertexPosition, 0, 1),
    instanceIndex: vertexIndex,
    uv: result.uv,
  };
});

console.log(tgpu.resolve({ externals: { mainVertex } }));

const mainFragment = tgpu['~unstable'].fragmentFn({
  in: {
    instanceIndex: d.interpolate('flat', d.u32),
    frontFacing: d.builtin.frontFacing,
    position: d.builtin.position,
    uv: d.vec2f,
  },
  out: d.vec4f,
})(({ position, instanceIndex, frontFacing, uv }) => {
  const colors = [
    d.vec3f(1, 0, 0), // 0
    d.vec3f(0, 1, 0), // 1
    d.vec3f(0, 0, 1), // 2
    d.vec3f(1, 0, 1), // 3
    d.vec3f(1, 1, 0), // 4
    d.vec3f(0, 1, 1), // 5
  ];
  const color = d.vec3f(0, cos(uv.y * 60), 1);
  // const color = colors[instanceIndex];
  if (frontFacing) {
    return d.vec4f(color, 0.5);
  }
  return d.vec4f(
    color,
    select(
      d.f32(0),
      d.f32(1),
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
  return d.vec4f(0, 0, 0, 0.2);
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

const centerlineVertex = tgpu['~unstable'].vertexFn({
  in: {
    vertexIndex: d.builtin.vertexIndex,
  },
  out: {
    outPos: d.builtin.position,
  },
})(({ vertexIndex }) => {
  const t = bindGroupLayout.$.uniforms.time;
  const v = bindGroupLayout.$.lineVertices[0].position;
  const vertex = animation(d.f32(vertexIndex), t, v.x, v.y);
  return {
    outPos: d.vec4f(vertex.position, 0, 1),
  };
});

const centerlinePipeline = root['~unstable']
  .withVertex(centerlineVertex, {})
  .withFragment(outlineFragment, {
    format: presentationFormat,
    blend: alphaBlend,
  })
  .withMultisample({ count: multisample ? 4 : 1 })
  .withPrimitive({
    topology: 'line-strip',
  })
  .createPipeline();

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
  const t = bindGroupLayout.$.uniforms.time;
  const v = bindGroupLayout.$.lineVertices[0].position;
  const vertex = animation(d.f32(instanceIndex), t, v.x, v.y);
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

const draw = (timeMs: number) => {
  uniformsBuffer.write({
    time: timeMs * 1e-3,
  });
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
    .drawIndexed(lineSegmentIndicesCapLevel2.length, 200);

  // outlinePipeline
  //   .with(bindGroupLayout, uniformsBindGroup)
  //   .withColorAttachment(colorAttachment)
  //   .drawIndexed(outlineIndicesCapLevel1.length, 200);

  // circlesPipeline
  //   .with(bindGroupLayout, uniformsBindGroup)
  //   .withColorAttachment(colorAttachment)
  //   .draw(CIRCLE_SEGMENT_COUNT + 1, 4);

  centerlinePipeline
    .with(bindGroupLayout, uniformsBindGroup)
    .withColorAttachment(colorAttachment)
    .draw(200);
};

const runAnimationFrame = (timeMs: number) => {
  draw(timeMs);
  requestAnimationFrame(runAnimationFrame);
};
runAnimationFrame(0);

canvas.addEventListener('pointermove', (ev) => {
  const rect = canvas.getBoundingClientRect();
  const mx = 2 * (ev.clientX - rect.left) / rect.width - 1;
  const my = 2 * (rect.bottom - ev.clientY) / rect.height - 1;
  lineVerticesBuffer.writePartial([{
    idx: 0,
    value: { position: d.vec2f(mx, my) },
  }]);
});

export function onCleanup() {
  root.destroy();
}
