import { createFft1d, decomposeWorkgroups } from "@typegpu/sort";
import type { TgpuRoot } from "typegpu";
import { arrayOf, f32 } from "typegpu/data";
import { L, N, S, WG } from "./constants.ts";
import {
  PluckCommand,
  embedKernel,
  embedLayout,
  extractKernel,
  extractLayout,
  reanchorKernel,
  reanchorLayout,
  superposeKernel,
  superposeLayout,
} from "./shaders.ts";

export function createDst(root: TgpuRoot) {
  const device = root.device;

  const fft = createFft1d(root, { n: L, numLines: 1 });

  const modeY0Buffer = root
    .createBuffer(arrayOf(f32, S * N))
    .$usage("storage");
  const modeV0Buffer = root
    .createBuffer(arrayOf(f32, S * N))
    .$usage("storage");
  modeY0Buffer.write(Array.from(new Float32Array(S * N)));
  modeV0Buffer.write(Array.from(new Float32Array(S * N)));

  const deltaModeBuffer = root
    .createBuffer(arrayOf(f32, N))
    .$usage("storage");

  const pluckCmdBuffer = root
    .createBuffer(PluckCommand, {
      stringIndex: 0,
      pluckU: 0.5,
      pluckHeight: 0,
      currentTime: 0,
    })
    .$usage("uniform");

  const embedPipeline = root.createComputePipeline({ compute: embedKernel });
  const extractPipeline = root.createComputePipeline({
    compute: extractKernel,
  });
  const superposePipeline = root.createComputePipeline({
    compute: superposeKernel,
  });
  const reanchorPipeline = root.createComputePipeline({
    compute: reanchorKernel,
  });

  const reanchorTimeBuffer = root.createBuffer(f32, 0).$usage("uniform");

  const embedBg = root.createBindGroup(embedLayout, {
    cmd: pluckCmdBuffer,
    fftInput: fft.input,
  });

  let extractBg = root.createBindGroup(extractLayout, {
    fftOutput: fft.output(),
    deltaModes: deltaModeBuffer,
  });

  let superposeBg: ReturnType<typeof root.createBindGroup> | null = null;
  let reanchorBg: ReturnType<typeof root.createBindGroup> | null = null;

  /**
   * Must be called after simulation is created, to bind the superpose kernel
   * to the simulation-owned stringParams and pluckState buffers.
   */
  // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types resolved at bind group creation
  function connect(deps: { stringParamsBuffer: any; pluckStateBuffer: any }) {
    superposeBg = root.createBindGroup(superposeLayout, {
      cmd: pluckCmdBuffer,
      // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
      stringParams: deps.stringParamsBuffer as any,
      // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
      pluckState: deps.pluckStateBuffer as any,
      deltaModes: deltaModeBuffer,
      modeY0: modeY0Buffer,
      modeV0: modeV0Buffer,
    });
    reanchorBg = root.createBindGroup(reanchorLayout, {
      currentTime: reanchorTimeBuffer,
      // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
      stringParams: deps.stringParamsBuffer as any,
      // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
      pluckState: deps.pluckStateBuffer as any,
      modeY0: modeY0Buffer,
      modeV0: modeV0Buffer,
    });
  }

  function pluck(
    stringIndex: number,
    pluckU: number,
    pluckHeight: number,
    currentTime: number,
  ) {
    if (!superposeBg) {
      throw new Error("dst.connect() must be called before pluck()");
    }

    pluckCmdBuffer.write({
      stringIndex,
      pluckU,
      pluckHeight,
      currentTime,
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();

    embedPipeline
      .with(pass)
      .with(embedBg)
      .dispatchWorkgroups(...decomposeWorkgroups(Math.ceil(L / WG)));

    fft.encodeForward(pass);

    extractBg = root.createBindGroup(extractLayout, {
      fftOutput: fft.output(),
      deltaModes: deltaModeBuffer,
    });
    extractPipeline
      .with(pass)
      .with(extractBg)
      .dispatchWorkgroups(...decomposeWorkgroups(Math.ceil(N / WG)));

    superposePipeline
      .with(pass)
      .with(superposeBg)
      .dispatchWorkgroups(...decomposeWorkgroups(Math.ceil(N / WG)));

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function reanchor(currentTime: number) {
    if (!reanchorBg) {
      throw new Error("dst.connect() must be called before reanchor()");
    }
    reanchorTimeBuffer.write(currentTime);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    reanchorPipeline
      .with(pass)
      .with(reanchorBg)
      .dispatchWorkgroups(Math.ceil(N / WG), S);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function destroy() {
    fft.destroy();
    modeY0Buffer.destroy();
    modeV0Buffer.destroy();
    deltaModeBuffer.destroy();
    pluckCmdBuffer.destroy();
    reanchorTimeBuffer.destroy();
  }

  return { modeY0Buffer, modeV0Buffer, connect, pluck, reanchor, destroy };
}
