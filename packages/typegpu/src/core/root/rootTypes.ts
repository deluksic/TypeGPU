import type { OmitBuiltins } from '../../builtin';
import type { AnyData, Disarray } from '../../data/dataTypes';
import type { Exotic } from '../../data/exotic';
import type { AnyWgslData, Vec4f, WgslArray } from '../../data/wgslTypes';
import type { JitTranspiler } from '../../jitTranspiler';
import type { NameRegistry } from '../../nameRegistry';
import type { Infer } from '../../shared/repr';
import type { Mutable, OmitProps, Prettify } from '../../shared/utilityTypes';
import type {
  LayoutEntryToInput,
  TgpuBindGroup,
  TgpuBindGroupLayout,
  TgpuLayoutEntry,
} from '../../tgpuBindGroupLayout';
import type { Unwrapper } from '../../unwrapper';
import type { TgpuBuffer, Vertex } from '../buffer/buffer';
import type { TgpuBufferUsage } from '../buffer/bufferUsage';
import type { IOLayout, IORecord } from '../function/fnTypes';
import type { TgpuComputeFn } from '../function/tgpuComputeFn';
import type { TgpuFn } from '../function/tgpuFn';
import type { TgpuFragmentFn } from '../function/tgpuFragmentFn';
import type { TgpuVertexFn } from '../function/tgpuVertexFn';
import type { TgpuComputePipeline } from '../pipeline/computePipeline';
import type {
  FragmentOutToTargets,
  TgpuRenderPipeline,
} from '../pipeline/renderPipeline';
import type { Eventual, TgpuAccessor, TgpuSlot } from '../slot/slotTypes';
import type { TgpuTexture } from '../texture/texture';
import type { LayoutToAllowedAttribs } from '../vertexLayout/vertexAttribute';
import type { TgpuVertexLayout } from '../vertexLayout/vertexLayout';

// ----------
// Public API
// ----------

export interface WithCompute {
  createPipeline(): TgpuComputePipeline;
}

export type ValidateFragmentIn<
  VertexOut extends IORecord,
  FragmentIn extends IORecord,
  FragmentOut extends IOLayout<Vec4f>,
> = FragmentIn extends Partial<VertexOut>
  ? VertexOut extends FragmentIn
    ? [
        entryFn: TgpuFragmentFn<FragmentIn, FragmentOut>,
        targets: FragmentOutToTargets<FragmentOut>,
      ]
    : [
        entryFn: 'n/a',
        targets: 'n/a',
        MissingFromVertexOutput: {
          [Key in Exclude<keyof FragmentIn, keyof VertexOut>]: FragmentIn[Key];
        },
      ]
  : [
      entryFn: 'n/a',
      targets: 'n/a',
      MismatchedVertexOutput: {
        [Key in keyof FragmentIn &
          keyof VertexOut as FragmentIn[Key] extends VertexOut[Key]
          ? never
          : Key]: [got: VertexOut[Key], expecting: FragmentIn[Key]];
      },
    ];

export interface WithVertex<VertexOut extends IORecord = IORecord> {
  withFragment<
    FragmentIn extends IORecord,
    FragmentOut extends IOLayout<Vec4f>,
  >(
    ...args: ValidateFragmentIn<VertexOut, FragmentIn, FragmentOut>
  ): WithFragment<FragmentOut>;
}

export interface WithFragment<
  Output extends IOLayout<Vec4f> = IOLayout<Vec4f>,
> {
  withPrimitive(
    primitiveState: GPUPrimitiveState | undefined,
  ): WithFragment<Output>;

  withDepthStencil(
    depthStencilState: GPUDepthStencilState | undefined,
  ): WithFragment<Output>;

  createPipeline(): TgpuRenderPipeline<Output>;
}

export interface WithBinding {
  with<T>(slot: TgpuSlot<T>, value: Eventual<T>): WithBinding;
  with<T extends AnyWgslData>(
    accessor: TgpuAccessor<T>,
    value: TgpuFn<[], T> | TgpuBufferUsage<T> | Infer<T>,
  ): WithBinding;

  withCompute(entryFn: TgpuComputeFn): WithCompute;

  withVertex<VertexIn extends IOLayout, VertexOut extends IORecord>(
    entryFn: TgpuVertexFn<VertexIn, VertexOut>,
    attribs: LayoutToAllowedAttribs<OmitBuiltins<VertexIn>>,
  ): WithVertex<VertexOut>;
}

export type CreateTextureOptions<
  TSize,
  TFormat extends GPUTextureFormat,
  TMipLevelCount extends number,
  TSampleCount extends number,
  TViewFormat extends GPUTextureFormat,
  TDimension extends GPUTextureDimension,
> = {
  /**
   * The width, height, and depth or layer count of the texture.
   */
  size: TSize;
  /**
   * The format of the texture.
   */
  format: TFormat;
  /**
   * The number of mip levels the texture will contain.
   * @default 1
   */
  mipLevelCount?: TMipLevelCount | undefined;
  /**
   * The sample count of the texture. A sampleCount > 1 indicates a multisampled texture.
   * @default 1
   */
  sampleCount?: TSampleCount | undefined;
  /**
   * Specifies extra formats (in addition to the texture's actual format) that can be used
   * when creating views of this texture.
   * @default []
   */
  viewFormats?: TViewFormat[] | undefined;
  /**
   * Whether the texture is one-dimensional, an array of two-dimensional layers, or three-dimensional.
   * @default '2d'
   */
  dimension?: TDimension | undefined;
};

export type CreateTextureResult<
  TSize extends readonly number[],
  TFormat extends GPUTextureFormat,
  TMipLevelCount extends number,
  TSampleCount extends number,
  TViewFormat extends GPUTextureFormat,
  TDimension extends GPUTextureDimension,
> = Prettify<
  {
    size: Mutable<TSize>;
    format: TFormat;
  } & OmitProps<
    {
      dimension: GPUTextureDimension extends TDimension
        ? // Omitted property means the default
          undefined
        : // '2d' is the default, omitting from type
          TDimension extends '2d'
          ? undefined
          : TDimension;
      mipLevelCount: number extends TMipLevelCount
        ? // Omitted property means the default
          undefined
        : // '1' is the default, omitting from type
          TMipLevelCount extends 1
          ? undefined
          : TMipLevelCount;
      sampleCount: number extends TSampleCount
        ? // Omitted property means the default
          undefined
        : // '1' is the default, omitting from type
          TSampleCount extends 1
          ? undefined
          : TSampleCount;
      viewFormats: GPUTextureFormat extends TViewFormat
        ? // Omitted property means the default
          undefined
        : // 'never[]' is the default, omitting from type
          TViewFormat[] extends never[]
          ? undefined
          : TViewFormat[];
    },
    undefined
  >
>;

/**
 * TODO: Implement flexible pipelines as per discussion:
 * https://github.com/software-mansion/TypeGPU/discussions/713
 */
interface RenderPass {
  setPipeline(): void;

  setBindGroup<Entries extends Record<string, TgpuLayoutEntry | null>>(
    bindGroupLayout: TgpuBindGroupLayout<Entries>,
    bindGroup: TgpuBindGroup<Entries>,
  ): void;

  setVertexBuffer<TData extends WgslArray | Disarray>(
    vertexLayout: TgpuVertexLayout<TData>,
    buffer: TgpuBuffer<TData> & Vertex,
  ): void;

  draw(
    vertexCount: number,
    instanceCount?: number | undefined,
    firstVertex?: number | undefined,
    firstInstance?: number | undefined,
  ): void;

  drawIndexed(
    indexCount: number,
    instanceCount?: number | undefined,
    firstIndex?: number | undefined,
    baseVertex?: number | undefined,
    firstInstance?: number | undefined,
  ): void;
}

export interface TgpuRoot extends Unwrapper {
  /**
   * The GPU device associated with this root.
   */
  readonly device: GPUDevice;

  /**
   * Allocates memory on the GPU, allows passing data between host and shader.
   *
   * @remarks
   * Typed wrapper around a GPUBuffer.
   *
   * @param typeSchema The type of data that this buffer will hold.
   * @param initial The initial value of the buffer. (optional)
   */
  createBuffer<TData extends AnyData>(
    typeSchema: TData,
    initial?: Infer<Exotic<TData>> | undefined,
  ): TgpuBuffer<Exotic<TData>>;

  /**
   * Allocates memory on the GPU, allows passing data between host and shader.
   *
   * @remarks
   * Typed wrapper around a GPUBuffer.
   *
   * @param typeSchema The type of data that this buffer will hold.
   * @param gpuBuffer A vanilla WebGPU buffer.
   */
  createBuffer<TData extends AnyData>(
    typeSchema: TData,
    gpuBuffer: GPUBuffer,
  ): TgpuBuffer<Exotic<TData>>;

  /**
   * Creates a group of resources that can be bound to a shader based on a specified layout.
   *
   * @remarks
   * Typed wrapper around a GPUBindGroup.
   *
   * @example
   * const fooLayout = tgpu.bindGroupLayout({
   *  foo: { uniform: d.vec3f },
   *  bar: { texture: 'float' },
   * });
   *
   * const fooBuffer = ...;
   * const barTexture = ...;
   *
   * const fooBindGroup = root.createBindGroup(fooLayout, {
   *  foo: fooBuffer,
   *  bar: barTexture,
   * });
   *
   * @param layout Layout describing the bind group to be created.
   * @param entries A record with values being the resources populating the bind group
   * and keys being their associated names, matching the layout keys.
   */
  createBindGroup<
    Entries extends Record<string, TgpuLayoutEntry | null> = Record<
      string,
      TgpuLayoutEntry | null
    >,
  >(
    layout: TgpuBindGroupLayout<Entries>,
    entries: {
      [K in keyof OmitProps<Entries, null>]: LayoutEntryToInput<Entries[K]>;
    },
  ): TgpuBindGroup<Entries>;

  /**
   * Destroys all underlying resources (i.e. buffers...) created through this root object.
   * If the object is created via `tgpu.init` instead of `tgpu.initFromDevice`,
   * then the inner GPU device is destroyed as well.
   */
  destroy(): void;

  '~unstable': Omit<ExperimentalTgpuRoot, keyof TgpuRoot>;
}

export interface ExperimentalTgpuRoot extends TgpuRoot, WithBinding {
  readonly jitTranspiler?: JitTranspiler | undefined;
  readonly nameRegistry: NameRegistry;
  /**
   * The current command encoder. This property will
   * hold the same value until `flush()` is called.
   */
  readonly commandEncoder: GPUCommandEncoder;

  createTexture<
    TWidth extends number,
    THeight extends number,
    TDepth extends number,
    TSize extends
      | readonly [TWidth]
      | readonly [TWidth, THeight]
      | readonly [TWidth, THeight, TDepth],
    TFormat extends GPUTextureFormat,
    TMipLevelCount extends number,
    TSampleCount extends number,
    TViewFormat extends GPUTextureFormat,
    TDimension extends GPUTextureDimension,
  >(
    props: CreateTextureOptions<
      TSize,
      TFormat,
      TMipLevelCount,
      TSampleCount,
      TViewFormat,
      TDimension
    >,
  ): TgpuTexture<
    CreateTextureResult<
      TSize,
      TFormat,
      TMipLevelCount,
      TSampleCount,
      TViewFormat,
      TDimension
    >
  >;

  /**
   * Causes all commands enqueued by pipelines to be
   * submitted to the GPU.
   */
  flush(): void;
}
