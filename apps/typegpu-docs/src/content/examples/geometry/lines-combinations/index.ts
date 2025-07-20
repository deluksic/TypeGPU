import tgpu from 'typegpu';
import type { ColorAttachment } from '../../../../../../../packages/typegpu/src/core/pipeline/renderPipeline.ts';
import { clamp, cos, min, mix, select, sin } from 'typegpu/std';
import {
  endCapSlot,
  joinSlot,
  lineCaps,
  lineJoins,
  lineSegmentIndicesCapLevel0,
  lineSegmentIndicesCapLevel3,
  lineSegmentVariableWidth,
  lineSegmentWireframeIndicesCapLevel0,
  lineSegmentWireframeIndicesCapLevel3,
  startCapSlot,
} from '@typegpu/geometry';
import { addMul } from '../../../../../../../packages/typegpu-geometry/src/utils.ts';
import {
  arrayOf,
  builtin,
  f32,
  interpolate,
  struct,
  u16,
  u32,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data';
import * as testCases from './testCases.ts';
import { TEST_SEGMENT_COUNT } from './constants.ts';

const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const canvas = document.querySelector('canvas');
const context = canvas?.getContext('webgpu');

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

const Uniforms = struct({
  time: f32,
  fillType: u32,
});

const uniformsBuffer = root.createBuffer(Uniforms, {
  time: 0,
  fillType: 1,
}).$usage('uniform');

const bindGroupLayout = tgpu.bindGroupLayout({
  uniforms: {
    uniform: Uniforms,
  },
});

const uniformsBindGroup = root.createBindGroup(bindGroupLayout, {
  uniforms: uniformsBuffer,
});

const indexBuffer = root.createBuffer(
  arrayOf(u16, lineSegmentIndicesCapLevel3.length),
  lineSegmentIndicesCapLevel3,
).$usage('index');

const outlineIndexBuffer = root.createBuffer(
  arrayOf(u16, lineSegmentWireframeIndicesCapLevel3.length),
  lineSegmentWireframeIndicesCapLevel3,
).$usage('index');

const testCaseSlot = tgpu.slot(testCases.arms);

const mainVertex = tgpu['~unstable'].vertexFn({
  in: {
    instanceIndex: builtin.instanceIndex,
    vertexIndex: builtin.vertexIndex,
  },
  out: {
    outPos: builtin.position,
    position: vec2f,
    instanceIndex: interpolate('flat', u32),
    vertexIndex: interpolate('flat', u32),
    situationIndex: interpolate('flat', u32),
  },
})(({ vertexIndex, instanceIndex }) => {
  const t = bindGroupLayout.$.uniforms.time;
  const A = testCaseSlot.$(instanceIndex, t);
  const B = testCaseSlot.$(instanceIndex + 1, t);
  const C = testCaseSlot.$(instanceIndex + 2, t);
  const D = testCaseSlot.$(instanceIndex + 3, t);

  // disconnect lines if radius is < 0
  if (A.radius < 0 || B.radius < 0 || C.radius < 0 || D.radius < 0) {
    return {
      outPos: vec4f(),
      position: vec2f(),
      instanceIndex: 0,
      vertexIndex: 0,
      situationIndex: 0,
    };
  }

  const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D);

  return {
    outPos: vec4f(result.vertexPosition, 0, 1),
    position: result.vertexPosition,
    instanceIndex,
    vertexIndex,
    situationIndex: result.situationIndex,
  };
});

console.log(tgpu.resolve({ externals: { lineSegmentVariableWidth } }));

const mainFragment = tgpu['~unstable'].fragmentFn({
  in: {
    instanceIndex: interpolate('flat', u32),
    vertexIndex: interpolate('flat', u32),
    situationIndex: interpolate('flat', u32),
    frontFacing: builtin.frontFacing,
    screenPosition: builtin.position,
    position: vec2f,
  },
  out: vec4f,
})(
  (
    {
      instanceIndex,
      vertexIndex,
      situationIndex,
      frontFacing,
      screenPosition,
      position,
    },
  ) => {
    const fillType = bindGroupLayout.$.uniforms.fillType;
    if (fillType === 1) {
      // typegpu gradient
      return mix(
        vec4f(0.77, 0.39, 1, 0.5),
        vec4f(0.11, 0.44, 0.94, 0.5),
        position.x * 0.5 + 0.5,
      );
    }
    let index = instanceIndex;
    if (fillType === 3) {
      index = vertexIndex;
    }
    if (fillType === 4) {
      index = situationIndex;
    }
    const colors = [
      vec3f(1, 0, 0), // 0
      vec3f(0, 1, 0), // 1
      vec3f(0, 0, 1), // 2
      vec3f(1, 0, 1), // 3
      vec3f(1, 1, 0), // 4
      vec3f(0, 1, 1), // 5
    ];
    const color = colors[index % 6];
    if (frontFacing) {
      return vec4f(color, 0.5);
    }
    return vec4f(
      color,
      select(
        f32(0),
        f32(1),
        ((u32(screenPosition.x) >> 3) % 2) !==
          ((u32(screenPosition.y) >> 3) % 2),
      ),
    );
  },
);

const centerlineVertex = tgpu['~unstable'].vertexFn({
  in: {
    vertexIndex: builtin.vertexIndex,
  },
  out: {
    outPos: builtin.position,
  },
})(({ vertexIndex }) => {
  const t = bindGroupLayout.$.uniforms.time;
  const vertex = testCaseSlot.$(vertexIndex, t);
  if (vertex.radius < 0) {
    return {
      outPos: vec4f(),
    };
  }
  return {
    outPos: vec4f(vertex.position, 0, 1),
  };
});

const outlineFragment = tgpu['~unstable'].fragmentFn({
  in: {
    _unused: builtin.frontFacing,
  },
  out: vec4f,
})(() => {
  return vec4f(0, 0, 0, 0.2);
});

const alphaBlend: GPUBlendState = {
  color: {
    operation: 'add',
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
  },
  alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
};

const CIRCLE_SEGMENT_COUNT = 256;
const CIRCLE_MIN_STEP = 2 * Math.PI / CIRCLE_SEGMENT_COUNT;
const CIRCLE_MAX_STEP = Math.PI / 8;
const CIRCLE_DASH_LEN = 0.0025 * Math.PI;

const circlesVertex = tgpu['~unstable'].vertexFn({
  in: {
    instanceIndex: builtin.instanceIndex,
    vertexIndex: builtin.vertexIndex,
  },
  out: {
    outPos: builtin.position,
  },
})(({ instanceIndex, vertexIndex }) => {
  const t = bindGroupLayout.$.uniforms.time;
  const vertex = testCaseSlot.$(instanceIndex, t);
  if (vertex.radius < 0) {
    return {
      outPos: vec4f(),
    };
  }
  const step = clamp(
    CIRCLE_DASH_LEN / vertex.radius,
    CIRCLE_MIN_STEP,
    CIRCLE_MAX_STEP,
  );
  const angle = min(2 * Math.PI, step * f32(vertexIndex));
  const unit = vec2f(cos(angle), sin(angle));
  return {
    outPos: vec4f(addMul(vertex.position, unit, vertex.radius), 0, 1),
  };
});

let testCase = testCases.arms;
let join = lineJoins.round;
let startCap = lineCaps.round;
let endCap = lineCaps.round;

function createPipelines() {
  const fill = root['~unstable']
    .with(joinSlot, join)
    .with(startCapSlot, startCap)
    .with(endCapSlot, endCap)
    .with(testCaseSlot, testCase)
    .withVertex(mainVertex, {})
    .withFragment(mainFragment, {
      format: presentationFormat,
      blend: alphaBlend,
    })
    .withPrimitive({
      // cullMode: 'back',
    })
    .createPipeline()
    .withIndexBuffer(indexBuffer);

  const outline = root['~unstable']
    .with(joinSlot, join)
    .with(startCapSlot, startCap)
    .with(endCapSlot, endCap)
    .with(testCaseSlot, testCase)
    .withVertex(mainVertex, {})
    .withFragment(outlineFragment, {
      format: presentationFormat,
      blend: alphaBlend,
    })
    .withPrimitive({
      topology: 'line-list',
    })
    .createPipeline()
    .withIndexBuffer(outlineIndexBuffer);

  const centerline = root['~unstable']
    .with(testCaseSlot, testCase)
    .withVertex(centerlineVertex, {})
    .withFragment(outlineFragment, {
      format: presentationFormat,
      blend: alphaBlend,
    })
    .withPrimitive({
      topology: 'line-strip',
    })
    .createPipeline();

  const circles = root['~unstable']
    .with(testCaseSlot, testCase)
    .withVertex(circlesVertex, {})
    .withFragment(outlineFragment, {
      format: presentationFormat,
      blend: alphaBlend,
    })
    .withPrimitive({
      topology: 'line-list',
    })
    .createPipeline();

  return {
    fill,
    outline,
    centerline,
    circles,
  };
}

let pipelines = createPipelines();

let showRadii = false;
let wireframe = true;
let fillType = 1;
let animationSpeed = 1;
let reverse = false;

const draw = (timeMs: number) => {
  uniformsBuffer.writePartial({
    time: timeMs * 1e-3,
  });
  const colorAttachment: ColorAttachment = {
    view: context.getCurrentTexture().createView(),
    clearValue: [1, 1, 1, 1],
    loadOp: 'load',
    storeOp: 'store',
  };
  pipelines.fill
    .with(bindGroupLayout, uniformsBindGroup)
    .withColorAttachment({ ...colorAttachment, loadOp: 'clear' })
    .withPerformanceCallback((start, end) => {
      if (frameId % 20 === 0) {
        console.log(`${(Number(end - start) * 1e-6).toFixed(2)} ms`);
      }
    })
    .drawIndexed(
      lineSegmentIndicesCapLevel3.length,
      fillType === 0 ? 0 : TEST_SEGMENT_COUNT,
    );

  if (wireframe) {
    pipelines.outline
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment(colorAttachment)
      .drawIndexed(
        lineSegmentWireframeIndicesCapLevel3.length,
        TEST_SEGMENT_COUNT,
      );
  }
  if (showRadii) {
    pipelines.circles
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment(colorAttachment)
      .draw(CIRCLE_SEGMENT_COUNT + 1, TEST_SEGMENT_COUNT);

    pipelines.centerline
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment(colorAttachment)
      .draw(TEST_SEGMENT_COUNT);
  }
};

let time = 0;
let lastFrameTime = 0;
let frameId = -1;
const runAnimationFrame = (timeMs: number) => {
  const deltaTime = timeMs - lastFrameTime;
  draw(time);
  frameId = requestAnimationFrame(runAnimationFrame);
  time += deltaTime * animationSpeed * (reverse ? -1 : 1);
  lastFrameTime = timeMs;
};
runAnimationFrame(0);

const fillOptions = {
  none: 0,
  solid: 1,
  instance: 2,
  triangle: 3,
  situation: 4,
};

export const controls = {
  'Test Case': {
    initial: Object.keys(testCases)[0],
    options: Object.keys(testCases),
    onSelectChange: async (selected: keyof typeof testCases) => {
      testCase = testCases[selected];
      pipelines = createPipelines();
    },
  },
  'Start Cap': {
    initial: 'round',
    options: Object.keys(lineCaps),
    onSelectChange: async (selected: keyof typeof lineCaps) => {
      startCap = lineCaps[selected];
      pipelines = createPipelines();
    },
  },
  'End Cap': {
    initial: 'round',
    options: Object.keys(lineCaps),
    onSelectChange: async (selected: keyof typeof lineCaps) => {
      endCap = lineCaps[selected];
      pipelines = createPipelines();
    },
  },
  'Join': {
    initial: 'round',
    options: Object.keys(lineJoins),
    onSelectChange: async (selected: keyof typeof lineJoins) => {
      join = lineJoins[selected];
      pipelines = createPipelines();
    },
  },
  'Fill': {
    initial: 'solid',
    options: Object.keys(fillOptions),
    onSelectChange: async (selected: keyof typeof fillOptions) => {
      fillType = fillOptions[selected];
      uniformsBuffer.writePartial({ fillType });
    },
  },
  'Wireframe': {
    initial: true,
    onToggleChange: (value: boolean) => {
      wireframe = value;
    },
  },
  'Radius and centerline': {
    initial: true,
    onToggleChange: (value: boolean) => {
      showRadii = value;
    },
  },
  'Animation speed': {
    initial: 1,
    min: 0,
    step: 0.001,
    max: 5,
    onSliderChange: (value: number) => {
      animationSpeed = value;
    },
  },
  'Reverse': {
    initial: false,
    onToggleChange: (value: boolean) => {
      reverse = value;
    },
  },
};

export function onCleanup() {
  root.destroy();
  cancelAnimationFrame(frameId);
}
