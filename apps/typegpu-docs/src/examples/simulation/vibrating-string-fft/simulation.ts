import type { TgpuRoot } from "typegpu";
import { arrayOf, f32 } from "typegpu/data";
import {
  DAMPER_STRENGTH,
  INTRA_FRAME_SAMPLES,
  N,
  S,
  SPATIAL_Y_FLOATS,
  WG,
  buildSinTable,
} from "./constants.ts";
import {
  GlobalUniforms,
  PluckState,
  StringParams,
  synthesizeLayout,
  synthesizeMultiKernel,
} from "./shaders.ts";

export type BaseParams = {
  fundamentalHz: number;
  stiffnessB: number;
  dampingGamma: number;
  spectralDamping: number;
  pickupU: number;
  damped: boolean;
};

// biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types resolved at bind group creation
export function createSimulation(root: TgpuRoot, deps: { modeY0Buffer: any; modeV0Buffer: any }) {
  const sinTable = buildSinTable();
  const sinTableBuffer = root
    .createBuffer(arrayOf(f32, N * N), [...sinTable])
    .$usage("storage");

  const stringParamsBuffer = root
    .createBuffer(arrayOf(StringParams, S))
    .$usage("storage");

  const pluckStateBuffer = root
    .createBuffer(arrayOf(PluckState, S))
    .$usage("storage");

  const globalUniformBuffer = root
    .createBuffer(GlobalUniforms, {
      frameTimeStart: 0,
      frameTimeEnd: 0,
      stringCount: S,
    })
    .$usage("uniform");

  const spatialYBuffer = root
    .createBuffer(arrayOf(f32, SPATIAL_Y_FLOATS))
    .$usage("storage");
  spatialYBuffer.write(Array.from(new Float32Array(SPATIAL_Y_FLOATS)));

  const pipeline = root.createComputePipeline({
    compute: synthesizeMultiKernel,
  });

  const bg = root.createBindGroup(synthesizeLayout, {
    globals: globalUniformBuffer,
    // biome-ignore lint/suspicious/noExplicitAny: buffer type inference across modules
    stringParams: stringParamsBuffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: buffer type inference across modules
    pluckState: pluckStateBuffer as any,
    sinTable: sinTableBuffer,
    // biome-ignore lint/suspicious/noExplicitAny: buffer type inference across modules
    modeY0: deps.modeY0Buffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: buffer type inference across modules
    modeV0: deps.modeV0Buffer as any,
    spatialY: spatialYBuffer,
  });

  function updateStringParams(base: BaseParams) {
    const data = [];
    for (let i = 0; i < S; i++) {
      data.push({
        omega0: base.fundamentalHz * 2 * Math.PI * 2 ** ((i * 5) / 12),
        stiffnessB: base.stiffnessB,
        dampingGamma: base.dampingGamma,
        spectralDamping: base.spectralDamping,
        pickupU: base.pickupU,
        damperStrength: base.damped ? DAMPER_STRENGTH : 0,
      });
    }
    stringParamsBuffer.write(data);
  }

  function update(
    pass: GPUComputePassEncoder,
    frameTimeStart: number,
    frameTimeEnd: number,
  ) {
    globalUniformBuffer.writePartial({
      frameTimeStart,
      frameTimeEnd,
      stringCount: S,
    });

    pipeline
      .with(pass)
      .with(bg)
      .dispatchWorkgroups(Math.ceil(N / WG), INTRA_FRAME_SAMPLES, S);
  }

  function destroy() {
    sinTableBuffer.destroy();
    stringParamsBuffer.destroy();
    pluckStateBuffer.destroy();
    globalUniformBuffer.destroy();
    spatialYBuffer.destroy();
  }

  return {
    spatialYBuffer,
    stringParamsBuffer,
    pluckStateBuffer,
    globalUniformBuffer,
    updateStringParams,
    update,
    destroy,
  };
}
