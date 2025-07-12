// deno-fmt-ignore
export const lineSegmentIndices = [
  0, 2, 1,
  1, 2, 3,
  2, 4, 3,
  3, 6, 1,
  6, 3, 8,
  5, 6, 7,
  7, 6, 8,
  8, 9, 7,
]

// deno-fmt-ignore
export const lineSegmentIndicesCapLevel1 = [
  ...lineSegmentIndices,
  10, 0, 1,
  11, 3, 4,
  12, 6, 5,
  13, 9, 8,
]

// deno-fmt-ignore
export const lineSegmentIndicesCapLevel2 = [
  ...lineSegmentIndicesCapLevel1,
  14, 0, 10,
  15, 10, 1,
  16, 3, 11,
  17, 11, 4,
  18, 12, 5,
  19, 6, 12,
  20, 13, 8,
  21, 9, 13,
];

// deno-fmt-ignore
export const lineSegmentWireframeIndices = [
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

// deno-fmt-ignore
export const lineSegmentWireframeIndicesCapLevel1 = [
  ...lineSegmentWireframeIndices,
  0, 10,
  1, 10,
  3, 11,
  4, 11,
  5, 12,
  6, 12,
  8, 13,
  9, 13,
]

// deno-fmt-ignore
export const lineSegmentWireframeIndicesCapLevel2 = [
  ...lineSegmentWireframeIndicesCapLevel1,
  0, 14,
  14, 10,
  10, 15,
  15, 1,
  3, 16,
  16, 11,
  11, 17,
  17, 4,
  5, 18,
  18, 12,
  12, 19,
  19, 6,
  8, 20,
  20, 13,
  13, 21,
  21, 9,
];
