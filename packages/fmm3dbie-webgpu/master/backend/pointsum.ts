/**
 * Dense Helmholtz point sums in single precision (step 4a):
 *
 *   pot(i) = sum_j K(x_i, y_j) q_j,   K = G_k (slp) or dG_k/dn_y (dlp),
 *
 * with sources, normals and targets uploaded once per handle. Pairs with
 * r = 0 are skipped.
 */

const WG = 128;
// Sources per dispatch, so that no single dispatch runs long enough to trip a
// GPU watchdog.
const CHUNK = 8192;
const KINDS: Record<string, number> = { slp: 0, dlp: 1 };

const WGSL = /* wgsl */ `
struct Params { ntarg: u32, j0: u32, j1: u32, kind: u32, zk: f32 }
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> targ: array<vec4f>;
// two vec4f per source: (x, y, z, 0), (nx, ny, nz, 0)
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
@group(0) @binding(3) var<storage, read> q: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> pot: array<vec2f>;
const OVER4PI: f32 = 0.07957747154594767;
const WG = ${WG}u;
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
// (cos x, sin x) to a few ulp; WGSL's builtins only promise 2^-11 absolute
// error. Cody-Waite reduction by pi/2, Cephes polynomials (as in shaders.ts).
fn sincos(x: f32) -> vec2f {
  let qn = round(x * 0.63661977236758134);
  var r = x - qn * 1.5703125;
  r = r - qn * 4.837512969970703125e-4;
  r = r - qn * 7.54978995489188216e-8;
  let z = r * r;
  let s = r + r * z * (-1.6666654611e-1 + z * (8.3321608736e-3 + z * (-1.9515295891e-4)));
  let c = 1.0 - 0.5 * z + z * z * (4.166664568298827e-2 + z * (-1.388731625493765e-3 + z * 2.443315711809948e-5));
  switch (i32(qn) & 3) {
    case 0: { return vec2f(c, s); }
    case 1: { return vec2f(-s, c); }
    case 2: { return vec2f(-c, -s); }
    default: { return vec2f(s, -c); }
  }
}
var<workgroup> tx: array<vec4f, WG>;
var<workgroup> tn: array<vec4f, WG>;
var<workgroup> tq: array<vec2f, WG>;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let i = gid.x;
  let valid = i < U.ntarg;
  var xt = vec3f(0.0);
  if (valid) { xt = targ[i].xyz; }
  let dlp = U.kind == 1u;
  var acc = vec2f(0.0);
  for (var base = U.j0; base < U.j1; base += WG) {
    let j = base + lid.x;
    if (j < U.j1) { tx[lid.x] = src[2u * j]; tn[lid.x] = src[2u * j + 1u]; tq[lid.x] = q[j]; }
    workgroupBarrier();
    let nl = min(WG, U.j1 - base);
    var part = vec2f(0.0);
    for (var s = 0u; s < nl; s++) {
      let d = xt - tx[s].xyz;
      let r = sqrt(dot(d, d));
      if (r > 0.0) {
        var g = sincos(U.zk * r) * (OVER4PI / r);
        if (dlp) { g = cmul(g, vec2f(1.0, -U.zk * r)) * (dot(d, tn[s].xyz) / (r * r)); }
        part += cmul(g, tq[s]);
      }
    }
    acc += part;
    workgroupBarrier();
  }
  if (valid) { pot[i] += acc; }
}
`;

const pipelines = new WeakMap<GPUDevice, GPUComputePipeline>();

function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelines.get(device);
  if (!p) {
    p = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: 'main' },
    });
    pipelines.set(device, p);
  }
  return p;
}

const STORAGE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export class PointSum {
  readonly ns: number;
  readonly nt: number;
  private buffers: GPUBuffer[];
  private uni: GPUBuffer;
  private q: GPUBuffer;
  private pot: GPUBuffer;
  private rb: GPUBuffer;
  private bg: GPUBindGroup;

  /** src: 8 floats per source (position, 0, normal, 0); targ: 4 floats per target. */
  constructor(
    private device: GPUDevice,
    private zk: number,
    src: Float32Array<ArrayBuffer>,
    targ: Float32Array<ArrayBuffer>,
  ) {
    this.ns = src.length / 8;
    this.nt = targ.length / 4;
    const buf = (bytes: number, usage: number) => device.createBuffer({ size: Math.max(16, bytes), usage });
    this.uni = buf(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const t = buf(targ.byteLength, STORAGE());
    const s = buf(src.byteLength, STORAGE());
    this.q = buf(8 * this.ns, STORAGE());
    this.pot = buf(8 * this.nt, STORAGE());
    this.rb = buf(8 * this.nt, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.buffers = [this.uni, t, s, this.q, this.pot, this.rb];
    device.queue.writeBuffer(t, 0, targ);
    device.queue.writeBuffer(s, 0, src);
    this.bg = device.createBindGroup({
      layout: getPipeline(device).getBindGroupLayout(0),
      entries: [this.uni, t, s, this.q, this.pot].map((b, k) => ({ binding: k, resource: { buffer: b } })),
    });
  }

  /** q: 2 floats (re, im) per source. Returns 2 floats per target. */
  async apply(kindName: string, q: Float32Array<ArrayBuffer>): Promise<Float32Array> {
    const kind = KINDS[kindName];
    if (kind === undefined) throw new Error(`fmm3dbie_gpu: unknown kind ${kindName}`);
    if (q.length !== 2 * this.ns) throw new Error(`fmm3dbie_gpu: q must have ${this.ns} entries`);
    const { device } = this;
    const p = getPipeline(device);
    device.queue.writeBuffer(this.q, 0, q);
    let enc = device.createCommandEncoder();
    enc.clearBuffer(this.pot);
    device.queue.submit([enc.finish()]);
    const params = new ArrayBuffer(32);
    const pu = new Uint32Array(params);
    const pf = new Float32Array(params);
    for (let j0 = 0; j0 < this.ns; j0 += CHUNK) {
      pu[0] = this.nt;
      pu[1] = j0;
      pu[2] = Math.min(this.ns, j0 + CHUNK);
      pu[3] = kind;
      pf[4] = this.zk;
      device.queue.writeBuffer(this.uni, 0, params);
      enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, this.bg);
      pass.dispatchWorkgroups(Math.ceil(this.nt / WG));
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.pot, 0, this.rb, 0, 8 * this.nt);
    device.queue.submit([enc.finish()]);
    await this.rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.rb.getMappedRange().slice(0, 8 * this.nt));
    this.rb.unmap();
    return out;
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
  }
}
