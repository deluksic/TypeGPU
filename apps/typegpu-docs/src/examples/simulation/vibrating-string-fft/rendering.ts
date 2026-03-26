import { lineSegmentIndices } from "@typegpu/geometry";
import type { TgpuRoot } from "typegpu";
import { d } from "typegpu";
import { arrayOf, u16 } from "typegpu/data";
import { fullScreenTriangle } from "typegpu/common";
import {
  HDR_COLOR_FORMAT,
  INTRA_FRAME_LAYER_ALPHA,
  INTRA_FRAME_SAMPLES,
  MAX_JOIN,
  N,
  S,
  SPATIAL_Y_FLOATS,
} from "./constants.ts";
import {
  LineVertexDrawUniform,
  RenderUniform,
  alphaBlend,
  caps,
  endCapSlot,
  lineBindLayout,
  lineFragment,
  lineVertex,
  presentFragment,
  presentLayout,
  startCapSlot,
  stringShadowFragment,
} from "./shaders.ts";

export function createRenderer(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  context: ReturnType<TgpuRoot["configureContext"]>,
  // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types resolved at bind group creation
  deps: { spatialYBuffer: any; stringParamsBuffer: any },
) {
  const device = root.device;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  const renderUniformBuffer = root
    .createBuffer(RenderUniform, {
      aspect: 1,
      yScale: 1 / (S + 1),
      trailLayerAlpha: 1,
      n: N,
      stringCount: S,
    })
    .$usage("uniform");

  const lineVertexDrawBuffer = root
    .createBuffer(LineVertexDrawUniform, {
      ndcShift: d.vec2f(0, 0),
      radiusScale: 1,
    })
    .$usage("uniform");

  const lineShadowVertexDrawBuffer = root
    .createBuffer(LineVertexDrawUniform, {
      ndcShift: d.vec2f(0, -0.5 / (S + 1)),
      radiusScale: 1.8,
    })
    .$usage("uniform");

  const indices = lineSegmentIndices(MAX_JOIN);
  const indexBuffer = root
    .createBuffer(arrayOf(u16, indices.length), indices)
    .$usage("index");

  const presentSampler = root["~unstable"].createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const linePipeline = root
    .with(startCapSlot, caps.round)
    .with(endCapSlot, caps.round)
    .createRenderPipeline({
      vertex: lineVertex,
      fragment: lineFragment,
      targets: { format: HDR_COLOR_FORMAT, blend: alphaBlend },
      multisample: { count: 4 },
    })
    .withIndexBuffer(indexBuffer);

  const stringShadowPipeline = root
    .with(startCapSlot, caps.round)
    .with(endCapSlot, caps.round)
    .createRenderPipeline({
      vertex: lineVertex,
      fragment: stringShadowFragment,
      targets: { format: HDR_COLOR_FORMAT, blend: alphaBlend },
      multisample: { count: 4 },
    })
    .withIndexBuffer(indexBuffer);

  const presentPipeline = root.createRenderPipeline({
    vertex: fullScreenTriangle,
    fragment: presentFragment,
    targets: { format: presentationFormat },
  });

  const lineBindGroup = root.createBindGroup(lineBindLayout, {
    uniforms: renderUniformBuffer,
    lineVertexDraw: lineVertexDrawBuffer,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    spatialY: deps.spatialYBuffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    stringParams: deps.stringParamsBuffer as any,
  });

  const lineShadowBindGroup = root.createBindGroup(lineBindLayout, {
    uniforms: renderUniformBuffer,
    lineVertexDraw: lineShadowVertexDrawBuffer,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    spatialY: deps.spatialYBuffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    stringParams: deps.stringParamsBuffer as any,
  });

  let hdrMsaaTex: GPUTexture | undefined;
  let hdrMsaaView: GPUTextureView | undefined;
  let hdrResolveTexture:
    | ReturnType<(typeof root)["~unstable"]["createTexture"]>
    | undefined;
  let hdrResolvePassView: GPUTextureView | undefined;
  let presentBindGroup: ReturnType<typeof root.createBindGroup> | undefined;
  let hdrTargetW = 0;
  let hdrTargetH = 0;

  function ensureHdrTargets() {
    const w = Math.max(1, canvas.width);
    const h = Math.max(1, canvas.height);
    if (
      hdrTargetW === w &&
      hdrTargetH === h &&
      hdrResolveTexture &&
      !hdrResolveTexture.destroyed
    ) {
      return;
    }
    hdrTargetW = w;
    hdrTargetH = h;
    hdrMsaaTex?.destroy();
    hdrResolveTexture?.destroy();
    hdrMsaaTex = device.createTexture({
      size: [w, h],
      format: HDR_COLOR_FORMAT,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    hdrMsaaView = hdrMsaaTex.createView();
    hdrResolveTexture = root["~unstable"]
      .createTexture({ size: [w, h], format: HDR_COLOR_FORMAT })
      .$usage("sampled", "render");
    hdrResolvePassView = root.unwrap(hdrResolveTexture).createView();
    presentBindGroup = root.createBindGroup(presentLayout, {
      // biome-ignore lint/suspicious/noExplicitAny: createView overload
      scene: (hdrResolveTexture as any).createView(d.texture2d(d.f32)),
      samp: presentSampler,
    });
  }

  function updateAspect() {
    const w = Math.max(1, canvas.width);
    const h = Math.max(1, canvas.height);
    renderUniformBuffer.writePartial({ aspect: w / h });
  }

  const resizeObserver = new ResizeObserver(() => {
    ensureHdrTargets();
    updateAspect();
  });
  resizeObserver.observe(canvas);
  ensureHdrTargets();
  updateAspect();

  function draw(encoder: GPUCommandEncoder) {
    ensureHdrTargets();
    const msaa = hdrMsaaView;
    const bg = presentBindGroup;
    const resolveView = hdrResolvePassView;
    if (!msaa || !bg || !resolveView) {
      return;
    }

    renderUniformBuffer.writePartial({
      trailLayerAlpha: INTRA_FRAME_LAYER_ALPHA,
    });
    const instanceCount = (N + 1) * INTRA_FRAME_SAMPLES * S;

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: msaa,
          resolveTarget: resolveView,
          clearValue: [0.97, 0.97, 0.98, 1],
          loadOp: "clear",
          storeOp: "discard",
        },
      ],
    });

    stringShadowPipeline
      .with(lineShadowBindGroup)
      .with(pass)
      .drawIndexed(indices.length, instanceCount);

    linePipeline
      .with(lineBindGroup)
      .with(pass)
      .drawIndexed(indices.length, instanceCount);

    pass.end();

    const passPresent = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    presentPipeline.with(bg).with(passPresent).draw(3);
    passPresent.end();
  }

  function setYScale(v: number) {
    renderUniformBuffer.writePartial({ yScale: v });
  }

  function destroy() {
    resizeObserver.disconnect();
    hdrMsaaTex?.destroy();
    hdrResolveTexture?.destroy();
  }

  return { draw, setYScale, destroy };
}
