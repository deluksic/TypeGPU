import {
  PLUCK_U_MAX,
  PLUCK_U_MIN,
  S,
  STRING_NDC_X_CAP,
  clampScalar,
} from "./constants.ts";

export type PluckCallback = (
  stringIndex: number,
  u: number,
  height: number,
) => void;

export function createInteraction(
  canvas: HTMLCanvasElement,
  callbacks: { onPluck: PluckCallback },
) {
  let lastStringIndex = -1;
  let pointerDown = false;

  function pointerToStringU(clientX: number): number {
    const rect = canvas.getBoundingClientRect();
    const tn = (clientX - rect.left) / Math.max(1e-6, rect.width);
    const ndcX = 2 * tn - 1;
    return clampScalar(0.5 * (1 + ndcX / STRING_NDC_X_CAP), 0, 1);
  }

  function pointerToPluckU(clientX: number): number {
    return clampScalar(pointerToStringU(clientX), PLUCK_U_MIN, PLUCK_U_MAX);
  }

  function pointerToStringIndex(clientY: number): number {
    const rect = canvas.getBoundingClientRect();
    const tn = (clientY - rect.top) / Math.max(1e-6, rect.height);
    const ndcY = 1 - 2 * tn;
    const spacing = 2 / (S + 1);
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < S; i++) {
      const yOff = 1 - spacing * (i + 1);
      const dist = Math.abs(ndcY - yOff);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }

  function pointerToPluckHeight(clientY: number): number {
    const rect = canvas.getBoundingClientRect();
    const tn = (clientY - rect.top) / Math.max(1e-6, rect.height);
    const ndcY = 1 - 2 * tn;
    const sIdx = pointerToStringIndex(clientY);
    const spacing = 2 / (S + 1);
    const stringNdcY = 1 - spacing * (sIdx + 1);
    const delta = ndcY - stringNdcY;
    return clampScalar(delta * 2, -0.65, 0.65);
  }

  function triggerPluck(clientX: number, clientY: number) {
    const sIdx = pointerToStringIndex(clientY);
    const u = pointerToPluckU(clientX);
    const h = pointerToPluckHeight(clientY);
    if (sIdx !== lastStringIndex) {
      lastStringIndex = sIdx;
      callbacks.onPluck(sIdx, u, h);
    }
  }

  function onPointerDown(e: PointerEvent) {
    pointerDown = true;
    lastStringIndex = -1;
    canvas.setPointerCapture(e.pointerId);
    triggerPluck(e.clientX, e.clientY);
  }

  function onPointerMove(e: PointerEvent) {
    if (!pointerDown) return;
    triggerPluck(e.clientX, e.clientY);
  }

  function onPointerUp(e: PointerEvent) {
    pointerDown = false;
    lastStringIndex = -1;
    canvas.releasePointerCapture(e.pointerId);
  }

  function onPointerCancel(e: PointerEvent) {
    pointerDown = false;
    lastStringIndex = -1;
    canvas.releasePointerCapture(e.pointerId);
  }

  const hasRawUpdate =
    typeof (canvas as unknown as Record<string, unknown>).onpointerrawupdate !==
    "undefined";
  const moveEvent = hasRawUpdate
    ? ("pointerrawupdate" as const)
    : ("pointermove" as const);

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener(moveEvent, onPointerMove as EventListener);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerCancel);
  canvas.addEventListener("lostpointercapture", () => {
    pointerDown = false;
    lastStringIndex = -1;
  });

  function destroy() {
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener(moveEvent, onPointerMove as EventListener);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointercancel", onPointerCancel);
  }

  return { destroy };
}
