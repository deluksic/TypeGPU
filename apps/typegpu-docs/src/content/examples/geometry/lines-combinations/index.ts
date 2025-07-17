import tgpu from 'typegpu';
import type { ColorAttachment } from '../../../../../../../packages/typegpu/src/core/pipeline/renderPipeline.ts';
import { clamp, cos, min, mix, select, sin } from 'typegpu/std';
import {
  lineSegmentIndicesCapLevel0,
  lineSegmentIndicesCapLevel2,
  lineSegmentVariableWidth,
  lineSegmentWireframeIndicesCapLevel0,
  lineSegmentWireframeIndicesCapLevel2,
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
  arrayOf(u16, lineSegmentIndicesCapLevel2.length),
  lineSegmentIndicesCapLevel2,
).$usage('index');

const outlineIndexBuffer = root.createBuffer(
  arrayOf(u16, lineSegmentWireframeIndicesCapLevel2.length),
  lineSegmentWireframeIndicesCapLevel2,
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
  let A = testCaseSlot.$(instanceIndex, t);
  const B = testCaseSlot.$(instanceIndex + 1, t);
  const C = testCaseSlot.$(instanceIndex + 2, t);
  let D = testCaseSlot.$(instanceIndex + 3, t);

  // disconnect or cap lines if radius is < 0
  if (B.radius < 0 || C.radius < 0) {
    return {
      outPos: vec4f(),
      position: vec2f(),
      instanceIndex: 0,
      vertexIndex: 0,
      situationIndex: 0,
    };
  }
  if (A.radius < 0) {
    A = B;
  }
  if (D.radius < 0) {
    D = C;
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

function createPipelines() {
  const fill = root['~unstable']
    .with(testCaseSlot, testCase)
    .withVertex(mainVertex, {})
    .withFragment(mainFragment, {
      format: presentationFormat,
      blend: alphaBlend,
    })
    .withPrimitive({
      cullMode: 'back',
    })
    .createPipeline()
    .withIndexBuffer(indexBuffer);

  const outline = root['~unstable']
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
const INSTANCE_COUNT = 1000;

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
  if (fillType !== 0) {
    pipelines.fill
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment({ ...colorAttachment, loadOp: 'clear' })
      .drawIndexed(lineSegmentIndicesCapLevel0.length, INSTANCE_COUNT);
  }

  if (wireframe) {
    pipelines.outline
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment({
        ...colorAttachment,
        loadOp: fillType === 0 ? 'clear' : 'load',
      })
      .drawIndexed(lineSegmentWireframeIndicesCapLevel0.length, INSTANCE_COUNT);
  }
  if (showRadii) {
    pipelines.circles
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment(colorAttachment)
      .draw(CIRCLE_SEGMENT_COUNT + 1, INSTANCE_COUNT);

    pipelines.centerline
      .with(bindGroupLayout, uniformsBindGroup)
      .withColorAttachment(colorAttachment)
      .draw(INSTANCE_COUNT);
  }
};

let time = 0;
let lastFrameTime = 0;
const runAnimationFrame = (timeMs: number) => {
  const deltaTime = timeMs - lastFrameTime;
  draw(time);
  requestAnimationFrame(runAnimationFrame);
  time += deltaTime * animationSpeed * (reverse ? -1 : 1);
  lastFrameTime = timeMs;
};
runAnimationFrame(0);

const fillOptions = {
  None: 0,
  Solid: 1,
  Instance: 2,
  Triangle: 3,
  Situation: 4,
};

export const controls = {
  'Test Case': {
    initial: Object.keys(testCases).at(-1),
    options: Object.keys(testCases),
    onSelectChange: async (selected: keyof typeof testCases) => {
      testCase = testCases[selected];
      pipelines = createPipelines();
    },
  },
  'Fill': {
    initial: 'Solid',
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
    initial: false,
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
}
