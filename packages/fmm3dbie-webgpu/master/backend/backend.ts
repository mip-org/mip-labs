/**
 * The GPU side of fmm3dbie_gpu: runs in numbl's GPU helper worker. build.mjs
 * bundles it into _fmm3dbie_gpu_backend.numbl.js, whose source the gateway
 * passes to gpu.load(). Each method takes (device, args) and returns
 * structured-cloneable data (see numbl src/gpuHelper.ts).
 *
 * Objects live behind numeric handles until freed:
 *   points  (PointSum)  dense far-field sums between arbitrary points
 *   surface (Operator)  the six layer potentials of fmm3dbie's rpcomb
 *                       formulations on a quad-patch surface, with near
 *                       quadrature (the POC's Operator, poc/gpu/operator.ts)
 */
import { PointSum } from './pointsum.ts';
import { Surface } from './poc/surface.ts';
import { Gpu } from './poc/gpu/device.ts';
import { Operator, type OperatorOptions } from './poc/gpu/operator.ts';

type Entry = { kind: 'points'; obj: PointSum } | { kind: 'surface'; obj: Operator; npts: number; surf: Surface };

const handles = new Map<number, Entry>();
let nextId = 1;
const gpus = new WeakMap<GPUDevice, Gpu>();

function gpuFor(device: GPUDevice): Gpu {
  let g = gpus.get(device);
  if (!g) {
    g = new Gpu(device);
    gpus.set(device, g);
  }
  return g;
}

function get(id: number, kind: Entry['kind']): Entry {
  const e = handles.get(id);
  if (!e) throw new Error(`fmm3dbie_gpu: invalid or freed handle ${id}`);
  if (e.kind !== kind) throw new Error(`fmm3dbie_gpu: handle ${id} is not a ${kind} handle`);
  return e;
}

function add(e: Entry): { id: number } {
  const id = nextId++;
  handles.set(id, e);
  return { id };
}

function destroy(e: Entry) {
  e.obj.destroy();
}

/** Column-major (12, npts) from MATLAB to the POC's row-major (12, npts). */
function transposeSrcvals(sv: Float64Array): Float64Array {
  const npts = sv.length / 12;
  const out = new Float64Array(sv.length);
  for (let i = 0; i < npts; i++) for (let c = 0; c < 12; c++) out[c * npts + i] = sv[12 * i + c];
  return out;
}

export const backend = {
  async points_create(device: GPUDevice, a: { zk: number; src: Float32Array<ArrayBuffer>; targ: Float32Array<ArrayBuffer> }) {
    return add({ kind: 'points', obj: new PointSum(device, a.zk, a.src, a.targ) });
  },

  async points_apply(_device: GPUDevice, a: { id: number; kind: string; q: Float32Array<ArrayBuffer> }) {
    const e = get(a.id, 'points');
    return { pot: await (e.obj as PointSum).apply(a.kind, a.q) };
  },

  /**
   * a.srcvals: (12, npts) column-major Float64Array, patches of type 11 and
   * one order. Sets up the near weights at the surface nodes.
   */
  async surface_create(
    device: GPUDevice,
    a: { norder: number; srcvals: Float64Array; zk: number; alpha: number; lam: number; ifinout: number; opts?: OperatorOptions },
  ) {
    const t0 = performance.now();
    const surf = new Surface(a.norder, transposeSrcvals(a.srcvals));
    // progress is reported after each paced chunk of near weights
    let tFirst = 0;
    let nchunks = 0;
    const progress = () => {
      if (nchunks++ === 0) tFirst = performance.now();
    };
    // Oversample to order p + 3, not the proof of concept's p + 2: Gauss-Legendre
    // grids of odd and even size share no points, so no oversampled node can
    // coincide with a surface node (both grids have the patch center when
    // p and p + 2 are even). The FMM far field and its near-smooth
    // subtraction sum over a target's own patch and need this.
    const op = new Operator(gpuFor(device), surf, a.zk, a.alpha, a.lam, a.ifinout, { progress, nover: a.norder + 3, ...(a.opts ?? {}) });
    const info = await op.setup();
    const t1 = performance.now();
    return {
      ...add({ kind: 'surface', obj: op, npts: surf.npts, surf }),
      info: { ...info, nchunks, secondsToFirstChunk: ((tFirst || t1) - t0) / 1000, secondsTotal: (t1 - t0) / 1000 },
    };
  },

  /**
   * One of the six layer potentials on the surface (principal value), with q
   * interleaved complex (2*npts): 0 S'_k, 1 S_kappa, 2 S'_kappa,
   * 3 D'_k - D'_kappa, 4 S_k, 5 D_k, kappa = i|k|.
   */
  async surface_apply(_device: GPUDevice, a: { id: number; m: number; q: Float64Array }) {
    const e = get(a.id, 'surface');
    if (!(a.m >= 0 && a.m < 6)) throw new Error(`fmm3dbie_gpu: kernel index must be 0..5`);
    if (a.q.length !== 2 * (e as { npts: number }).npts) throw new Error(`fmm3dbie_gpu: q must have ${(e as { npts: number }).npts} entries`);
    return { pot: await (e.obj as Operator).applySingle(a.m, a.q) };
  },

  /**
   * One pass of fmm3dbie's helm_rpcomb_imp operator at the surface nodes:
   * the selected kernels (sel, 6 entries in {0, 1: qa, 2: qb}) applied to the
   * densities qa, qb (interleaved complex, npts), with near quadrature reqs
   * ([kernel, density] pairs). far = 'dense': far field by the dense GPU sum;
   * far = 'none': near quadrature only, for a far field computed on the
   * numbl side (an FMM). Returns F, (6, npts) interleaved complex, kernel-major.
   */
  async surface_applyAt(
    _device: GPUDevice,
    a: { id: number; qa: Float64Array; qb: Float64Array | null; sel: number[]; reqs: [number, number][]; far: 'dense' | 'none' },
  ) {
    const e = get(a.id, 'surface') as { obj: Operator; npts: number };
    if (a.qa.length !== 2 * e.npts || (a.qb && a.qb.length !== 2 * e.npts)) throw new Error(`fmm3dbie_gpu: densities must have ${e.npts} entries`);
    if (a.far !== 'dense' && a.far !== 'none') throw new Error(`fmm3dbie_gpu: far must be 'dense' or 'none'`);
    return { F: await e.obj.applyAtSurface(a.qa, a.qb, a.sel, a.reqs, a.far) };
  },

  /** The oversampled far-field sources and the oversampling matrix (see Operator.farGeometry). */
  async surface_farGeometry(_device: GPUDevice, a: { id: number }) {
    const e = get(a.id, 'surface') as { obj: Operator };
    return e.obj.farGeometry();
  },

  /**
   * What the solve on the numbl side needs: the weights of fmm3dbie's inner
   * product, the identity term zid, the operator's parameters and the nodes.
   */
  async surface_info(_device: GPUDevice, a: { id: number }) {
    const e = get(a.id, 'surface') as { obj: Operator; npts: number; surf: Surface };
    const op = e.obj;
    // srcvals: (12, npts) row-major (xyz, r_u, r_v, unit normal)
    return { wts: Float64Array.from(e.surf.wts), zid: op.zid(), k: op.k, alpha: op.alpha, ifinout: op.ifinout, srcvals: e.surf.srcvals };
  },

  /** phi1 = S_kappa[sigma] on the surface, kappa = i|k|. */
  async surface_sKappa(_device: GPUDevice, a: { id: number; sigma: Float64Array }) {
    const e = get(a.id, 'surface') as { obj: Operator; npts: number };
    if (a.sigma.length !== 2 * e.npts) throw new Error(`fmm3dbie_gpu: sigma must have ${e.npts} entries`);
    return { phi1: await e.obj.sKappa(a.sigma) };
  },

  /**
   * u = S_k[sigma] + i alpha D_k[phi1] at off-surface targets x, (3, n)
   * column-major, with near quadrature at the targets.
   */
  async surface_eval(_device: GPUDevice, a: { id: number; targ: Float64Array; sigma: Float64Array; phi1: Float64Array }) {
    const e = get(a.id, 'surface') as { obj: Operator; npts: number };
    if (a.sigma.length !== 2 * e.npts || a.phi1.length !== 2 * e.npts) throw new Error(`fmm3dbie_gpu: densities must have ${e.npts} entries`);
    const n = a.targ.length / 3;
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) x[d * n + i] = a.targ[3 * i + d];
    return { u: await e.obj.eval(x, a.sigma, a.phi1) };
  },

  async free(_device: GPUDevice, a: { id: number }) {
    const e = handles.get(a.id);
    if (e) {
      destroy(e);
      handles.delete(a.id);
    }
    return null;
  },

  async freeall() {
    handles.forEach(destroy);
    handles.clear();
    return null;
  },

  async count() {
    return handles.size;
  },

  dispose() {
    handles.forEach(destroy);
    handles.clear();
  },
};
