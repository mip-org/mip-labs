/** The GPU device, plus small helpers for buffers, uniforms, readback and paced dispatching. */

export async function requestSolverDevice(): Promise<Gpu> {
  if (!navigator.gpu) {
    throw new Error(
      'this browser has no WebGPU. Chrome and Edge 113+, Safari 26+, and Firefox 141+ on Windows have it; ' +
        'on Linux, Chrome may need chrome://flags/#enable-unsafe-webgpu and Firefox dom.webgpu.enabled.',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU found no adapter on this machine.');
  const lim = adapter.limits;
  // The near weights of a medium-sized problem are a few hundred MB, so ask
  // for the largest buffers the adapter will give rather than the 128 MB default.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxComputeWorkgroupStorageSize: Math.max(16384, Math.min(lim.maxComputeWorkgroupStorageSize, 32768)),
    },
  });
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') console.error(`WebGPU device lost: ${info.message}`);
  });
  return new Gpu(device, adapter);
}

export function adapterDescription(adapter: GPUAdapter | null): string {
  if (!adapter) return 'no adapter';
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  if (!info) return 'unknown adapter';
  return [info.description, info.device, info.architecture, info.vendor].filter((s) => s).join(', ') || 'unknown adapter';
}

/** A device wrapper with the conveniences the solver needs. */
export class Gpu {
  private pipelines = new Map<string, GPUComputePipeline>();

  /**
   * The adapter is held for as long as the device is used: in Chrome, async
   * device calls fail ("a valid external Instance reference no longer exists")
   * once the adapter has been garbage collected.
   */
  constructor(
    readonly device: GPUDevice,
    readonly adapter?: GPUAdapter,
  ) {}

  storage(data: ArrayBufferView | number, extraUsage: GPUBufferUsageFlags = 0): GPUBuffer {
    const size = typeof data === 'number' ? data : data.byteLength;
    const buf = this.device.createBuffer({
      size: Math.max(16, Math.ceil(size / 16) * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
    });
    if (typeof data !== 'number') this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    return buf;
  }

  uniform(bytes: number): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  write(buf: GPUBuffer, data: ArrayBufferView, offset = 0) {
    this.device.queue.writeBuffer(buf, offset, data.buffer, data.byteOffset, data.byteLength);
  }

  /** Compile a pipeline once; throws with the WGSL diagnostics if the shader does not compile. */
  async pipeline(key: string, code: string, entry: string): Promise<GPUComputePipeline> {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code, label: key });
      try {
        p = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: entry }, label: key });
      } catch (e) {
        throw new Error(`could not compile the ${key} kernel: ${e instanceof Error ? e.message : String(e)}`);
      }
      this.pipelines.set(key, p);
    }
    return p;
  }

  bind(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }

  dispatch(pipeline: GPUComputePipeline, group: GPUBindGroup, nGroups: number) {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(nGroups);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  clear(buf: GPUBuffer) {
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(buf);
    this.device.queue.submit([enc.finish()]);
  }

  async read(buf: GPUBuffer, bytes = buf.size): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
  }

  async sync() {
    await this.device.queue.onSubmittedWorkDone();
  }
}

/** Little-endian writer for uniform structs of 32-bit scalars. */
export class UniformWriter {
  private view: DataView;
  private off = 0;
  readonly bytes: ArrayBuffer;

  constructor(nWords: number) {
    this.bytes = new ArrayBuffer(Math.ceil((4 * nWords) / 16) * 16);
    this.view = new DataView(this.bytes);
  }
  u32(v: number) {
    this.view.setUint32(this.off, v >>> 0, true);
    this.off += 4;
    return this;
  }
  i32(v: number) {
    this.view.setInt32(this.off, v | 0, true);
    this.off += 4;
    return this;
  }
  f32(v: number) {
    this.view.setFloat32(this.off, v, true);
    this.off += 4;
    return this;
  }
  /** Skip to the next 16-byte boundary (vec4 alignment). */
  align16() {
    this.off = Math.ceil(this.off / 16) * 16;
    return this;
  }
}

/**
 * Chooses how much work to submit per dispatch so that each one takes about
 * `target` seconds: long enough to keep the GPU busy, short enough that the
 * page stays responsive, progress can be reported, and no driver watchdog is
 * tripped. Work is measured in arbitrary units chosen by the caller.
 */
export class Pacer {
  private rate = 0; // units per second, 0 = unknown

  constructor(
    private target = 0.15,
    private initial = 1,
  ) {}

  /** Units to submit next, given a hard cap. */
  next(cap: number): number {
    if (this.rate <= 0) return Math.min(cap, this.initial);
    return Math.max(1, Math.min(cap, Math.floor(this.rate * this.target)));
  }

  /** Record that `units` took `seconds`. */
  record(units: number, seconds: number) {
    if (seconds <= 0) return;
    const r = units / Math.max(seconds, 1e-3);
    this.rate = this.rate > 0 ? 0.5 * (this.rate + r) : r;
  }
}
