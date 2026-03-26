import type { TgpuRoot } from "typegpu";
import { arrayOf, f32 } from "typegpu/data";
import { AUDIO_SAMPLES_PER_WG, CHUNK_SAMPLES, LOOKAHEAD_SEC, S, SAMPLE_RATE } from "./constants.ts";
import {
  AudioUniforms,
  PluckState,
  StringParams,
  audioKernel,
  audioLayout,
} from "./shaders.ts";

export function createAudioManager(
  root: TgpuRoot,
  // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types resolved at bind group creation
  deps: { modeY0Buffer: any; modeV0Buffer: any; stringParamsBuffer: any; pluckStateBuffer: any },
) {
  const device = root.device;

  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const masterGain = audioCtx.createGain();
  masterGain.gain.value = 1;
  masterGain.connect(audioCtx.destination);

  let sessionGain = audioCtx.createGain();
  sessionGain.gain.value = 1;
  sessionGain.connect(masterGain);

  let audioOutputGain = 1.75;
  let nextScheduleTime = 0;
  let audioTimeCursor = 0;
  let running = false;
  let pumpChain: Promise<void> = Promise.resolve();

  const audioChunkBuffer = root
    .createBuffer(arrayOf(f32, CHUNK_SAMPLES))
    .$usage("storage");

  const audioUniformBuffer = root
    .createBuffer(AudioUniforms, {
      chunkStartTime: 0,
      invSampleRate: 1 / audioCtx.sampleRate,
      chunkSampleCount: CHUNK_SAMPLES,
      stringCount: S,
    })
    .$usage("uniform");

  const audioPipeline = root.createComputePipeline({ compute: audioKernel });

  const audioBg = root.createBindGroup(audioLayout, {
    uniforms: audioUniformBuffer,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    stringParams: deps.stringParamsBuffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    pluckState: deps.pluckStateBuffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    modeY0: deps.modeY0Buffer as any,
    // biome-ignore lint/suspicious/noExplicitAny: cross-module buffer types
    modeV0: deps.modeV0Buffer as any,
    audioOut: audioChunkBuffer,
  });

  function startSession(wallTime: number) {
    const now = audioCtx.currentTime;
    sessionGain.gain.cancelScheduledValues(now);
    sessionGain.gain.setValueAtTime(sessionGain.gain.value, now);
    sessionGain.gain.linearRampToValueAtTime(0, now + 0.004);
    const old = sessionGain;
    setTimeout(() => {
      old.disconnect();
    }, 50);

    sessionGain = audioCtx.createGain();
    sessionGain.gain.value = 0;
    sessionGain.connect(masterGain);
    nextScheduleTime = now + 0.006;
    audioTimeCursor = wallTime;

    sessionGain.gain.cancelScheduledValues(now);
    sessionGain.gain.setValueAtTime(0, now);
    sessionGain.gain.linearRampToValueAtTime(1, now + 0.02);

    running = true;
  }

  async function pumpOnce() {
    if (!running) {
      return;
    }
    let guard = 0;
    while (guard < 8) {
      guard++;
      const now = audioCtx.currentTime;
      if (nextScheduleTime >= now + LOOKAHEAD_SEC) {
        break;
      }

      audioUniformBuffer.writePartial({
        chunkStartTime: audioTimeCursor,
        chunkSampleCount: CHUNK_SAMPLES,
      });

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      audioPipeline
        .with(pass)
        .with(audioBg)
        .dispatchWorkgroups(
          Math.ceil(CHUNK_SAMPLES / AUDIO_SAMPLES_PER_WG),
        );
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const raw = await audioChunkBuffer.read();
      const arr = raw as unknown as number[] | Float32Array;
      const tmp = new Float32Array(CHUNK_SAMPLES);
      if (arr instanceof Float32Array) {
        tmp.set(arr.subarray(0, CHUNK_SAMPLES));
      } else {
        for (let i = 0; i < CHUNK_SAMPLES; i++) {
          tmp[i] = (arr[i] as number) ?? 0;
        }
      }
      const g = audioOutputGain;
      for (let i = 0; i < CHUNK_SAMPLES; i++) {
        const x = tmp[i] * g;
        tmp[i] = x <= -1 ? -1 : x >= 1 ? 1 : x;
      }

      const sr = audioCtx.sampleRate;
      const buf = audioCtx.createBuffer(1, CHUNK_SAMPLES, sr);
      buf.copyToChannel(tmp, 0, 0);
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(sessionGain);
      const tClock = audioCtx.currentTime;
      const t0 = Math.max(nextScheduleTime, tClock + 0.002);
      src.start(t0);
      nextScheduleTime = t0 + CHUNK_SAMPLES / sr;
      audioTimeCursor += CHUNK_SAMPLES / sr;
    }
  }

  function pump() {
    pumpChain = pumpChain.catch(() => {}).then(() => pumpOnce());
  }

  function resume() {
    void audioCtx.resume();
  }

  function setOutputGain(g: number) {
    audioOutputGain = g;
  }

  function destroy() {
    running = false;
    void audioCtx.close();
    audioChunkBuffer.destroy();
    audioUniformBuffer.destroy();
  }

  return { audioCtx, resume, startSession, pump, setOutputGain, destroy };
}
