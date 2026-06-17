import { struct, vec3f } from 'typegpu/data';

/**
 * Vertex shader output for procedural 3D patch shapes.
 *
 * Procedural shapes are drawn with **patch instancing**: each logical object uses
 * multiple GPU draw instances (one per surface patch — e.g. 12 for a cubesphere,
 * 44 for a rounded box). Pass `@builtin(instance_index)` into the shape function;
 * it selects the patch via `shapePatchIndex(instanceIndex)`.
 *
 * Look up per-object parameters **before** calling the shape function, using
 * `shapeObjectIndex(instanceIndex)` to index storage (`boxes[i]`, `spheres[i]`, …).
 *
 * All patches of a shape share one `segmentTriangleIndices(maxSegmentCount)` index buffer.
 * Draw only a prefix of that buffer per patch to set the live segment count;
 * finer prebuilt triangles are left out of the draw, throwing away resolution that
 * isn't needed, so segment count can change at runtime without rebuilding indices.
 */
export const ProceduralShapeResult = struct({
  vertex: vec3f,
  normal: vec3f,
});
