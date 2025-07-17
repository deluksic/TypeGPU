// deno-fmt-ignore
export const lineSegmentIndicesCapLevel0 = [
  0, 4, 5,
  1, 2, 0,
  2, 3, 4,
  4, 0, 2,
  5, 9, 0,
  6, 7, 5,
  7, 8, 9,
  9, 5, 7,
]

// deno-fmt-ignore
export const lineSegmentIndicesCapLevel1 = [
  ...lineSegmentIndicesCapLevel0,
  10, 1, 0,
  11, 4, 3,
  12, 6, 5,
  13, 9, 8,
]

// deno-fmt-ignore
export const lineSegmentIndicesCapLevel2 = [
  ...lineSegmentIndicesCapLevel1,
  14, 10, 0,
  15, 1, 10,
  16, 11, 3,
  17, 4, 11,
  18, 12, 5,
  19, 6, 12,
  20, 13, 8,
  21, 9, 13,
];

// deno-fmt-ignore
export const lineSegmentWireframeIndicesCapLevel0 = [
  0, 1,
  0, 2,
  0, 4,
  0, 5,
  0, 9,
  1, 2,
  2, 3,
  2, 4,
  3, 4,
  4, 5,
  5, 6,
  5, 7,
  5, 9,
  6, 7,
  7, 8,
  7, 9,
  8, 9,
];

// deno-fmt-ignore
export const lineSegmentWireframeIndicesCapLevel1 = [
  ...lineSegmentWireframeIndicesCapLevel0,
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
