/**
 * The discretized operator of fmm3dbie's right-preconditioned combined field
 * impedance equation (helm_rpcomb_imp), on WebGPU. A port of
 * fmm3dbie_gpu/solver.py from the CUDA proof of concept.
 *
 *     u = S_k[sigma] + i alpha D_k S_kappa[sigma],   kappa = i|k|,
 *
 * with the boundary condition du/dn + i k lambda u = f. The discretized
 * equation is
 *
 *     z sigma + S'_k sigma + i alpha S'_kappa phi2 + i alpha (D'_k - D'_kappa) phi1
 *       + i k lambda [S_k sigma + i alpha D_k phi1 + i alpha w phi1] = f,
 *
 *     phi1 = S_kappa sigma, phi2 = S'_kappa sigma,
 *     z = (-1)^ifinout / 2 - i alpha / 4,   w = 1/2 (exterior), -1/2 (interior).
 *
 * Every layer potential is the sum of a far part (smooth rule on the
 * oversampled nodes of the patches that are not near the target, the dense
 * kernel) and a near part (precomputed weights for the near patches, the
 * sparse kernel). The near weights are computed once, in chunks of pairs, and
 * kept on the GPU. GMRES runs on the CPU in double precision and calls
 * applyK for each Krylov vector.
 */
import { leggauss } from '../gauss.ts';
import { oversampleMatrix, type Surface } from '../surface.ts';
import type { CVec } from '../problem.ts';
import { gmres, type GmresResult } from '../gmres.ts';
import { Gpu, Pacer, UniformWriter } from './device.ts';
import { BUILD_LEAVES, DENSE_APPLY, DENSE_WG, NEAR_WEIGHTS, NT, SPARSE_APPLY } from './shaders.ts';

export interface OperatorOptions {
  /** near radius factor: a patch is near a target within rfac * (patch radius) of its centroid */
  rfac?: number;
  rfacQ?: number;
  nq?: number;
  nqr?: number;
  nqx?: number;
  subLen?: number;
  oscLen?: number;
  kRMax?: number;
  nlmax?: number;
  nover?: number;
  /** seconds of GPU work per dispatch */
  pace?: number;
  progress?: (stage: string, fraction: number) => void;
}

export interface NearPairs {
  pairT: Int32Array;
  pairP: Int32Array;
  rowPtr: Uint32Array;
  /**
   * per pair, 8 floats: (u_t, v_t, kind, 0, e_x, e_y, e_z, 0). kind 0: the displacement to
   * the target is a plain difference (off-surface targets, or on-surface targets that do
   * not map into the patch's chart); 1: an on-surface target with preimage (u_t, v_t) in
   * the patch's chart and gap e = chart(u_t, v_t) - x_t; 2: the target's own patch.
   */
  pairF: Float32Array;
  npairs: number;
}

/** A set of targets with everything needed to apply the operator at them. */
interface TargetSet {
  n: number;
  targ: GPUBuffer; // 2 vec4 per target: [x, 0], [n, 0]
  pairs: NearPairs;
  pairsI: GPUBuffer;
  pairsF: GPUBuffer;
  rowPtr: GPUBuffer;
  pairP: GPUBuffer;
  W: GPUBuffer;
  nleaves: number;
}

export interface SetupInfo {
  npts: number;
  noversampled: number;
  npairs: number;
  nleaves: number;
  weightsBytes: number;
  setupSeconds: number;
}

const now = () => performance.now() / 1000;

export class Operator {
  readonly k: number;
  readonly alpha: number;
  /** wall impedance; the near weights do not depend on it, so it may be changed between solves */
  lam: number;
  readonly ifinout: number;
  readonly norder: number;
  readonly npols: number;
  readonly npts: number;
  readonly nover: number;
  readonly npolsO: number;
  readonly rfac: number;
  readonly quad: { nq: number; nqr: number; nqx: number; subLen: number; oscLen: number };
  readonly leaf: { rfacQ: number; nlmax: number; kRMax: number };
  readonly progress: (stage: string, fraction: number) => void;
  readonly pace: number;
  info!: SetupInfo;

  private gpu: Gpu;
  private surf: Surface;
  private r2: Float64Array;
  private overMat: Float64Array;
  private gpar!: GPUBuffer;
  private tables!: GPUBuffer;
  private srcgeo!: GPUBuffer;
  private srcq!: GPUBuffer;
  private qbuf!: GPUBuffer;
  private out!: GPUBuffer;
  private wall!: TargetSet;
  private nself = 0;
  private selfItems: Float32Array[] = []; // per surface node, its polar segments (8 floats each)
  private uniformLeaves: GPUBuffer;
  private uniformWeights: GPUBuffer;
  private uniformDense: GPUBuffer;
  private uniformSparse: GPUBuffer;
  private destroyed = false;

  constructor(gpu: Gpu, surf: Surface, k: number, alpha: number, lam: number, ifinout = 0, opts: OperatorOptions = {}) {
    this.gpu = gpu;
    this.surf = surf;
    this.k = k;
    this.alpha = alpha;
    this.lam = lam;
    this.ifinout = ifinout;
    this.norder = surf.norder;
    this.npols = surf.npols;
    this.npts = surf.npts;
    if (this.norder > 10) throw new Error('order at most 10');
    this.rfac = opts.rfac ?? (this.norder <= 6 ? 2.0 : 1.25); // fmm3dbie's get_rfacs
    this.nover = opts.nover ?? this.norder + 2;
    this.npolsO = (this.nover + 1) ** 2;
    let radMax = 0;
    for (let j = 0; j < surf.npatches; j++) radMax = Math.max(radMax, surf.rads[j]);
    // the radial rule must resolve the basis and exp(ikr) along a ray
    const nqr = opts.nqr ?? Math.floor(Math.min(Math.max(12, this.norder + 2 * Math.abs(k) * radMax + 6), 32));
    this.quad = { nq: opts.nq ?? 8, nqr, nqx: opts.nqx ?? 8, subLen: opts.subLen ?? 1.0, oscLen: opts.oscLen ?? 2.0 };
    this.leaf = { rfacQ: opts.rfacQ ?? 2.0, nlmax: opts.nlmax ?? 14, kRMax: opts.kRMax ?? 3.0 };
    this.progress = opts.progress ?? (() => {});
    this.pace = opts.pace ?? 0.15;
    this.r2 = new Float64Array(surf.npatches);
    for (let j = 0; j < surf.npatches; j++) this.r2[j] = (this.rfac * surf.rads[j]) ** 2;
    this.overMat = oversampleMatrix(this.norder, this.nover);
    this.uniformLeaves = gpu.uniform(64);
    this.uniformWeights = gpu.uniform(64);
    this.uniformDense = gpu.uniform(64);
    this.uniformSparse = gpu.uniform(64);
  }

  /** Upload the geometry and compute the near weights for the surface nodes. */
  async setup(): Promise<SetupInfo> {
    const t0 = now();
    const { gpu, surf, npols, npts } = this;
    let gp: Float32Array;
    if (surf.exact) {
      // superellipsoid face charts: 4 vec4 per patch, [n, s0], [e1, s1], [e2, t0], [0, t1]
      gp = new Float32Array(16 * surf.npatches);
      for (let j = 0; j < surf.npatches; j++) {
        const par = surf.exact.params.subarray(13 * j, 13 * j + 13);
        gp.set([par[0], par[1], par[2], par[9], par[3], par[4], par[5], par[10], par[6], par[7], par[8], par[11], 0, 0, 0, par[12]], 16 * j);
      }
    } else {
      // Legendre coefficients of xyz, r_u, r_v: 9 * npols floats per patch, padded to vec4
      gp = new Float32Array(4 * Math.ceil((9 * npts) / 4));
      for (let j = 0; j < surf.npatches; j++) {
        for (let c = 0; c < 9; c++) {
          for (let n = 0; n < npols; n++) gp[j * 9 * npols + c * npols + n] = surf.srccoefs[c * npts + j * npols + n];
        }
      }
    }
    this.gpar = gpu.storage(gp);
    // umat and the quadrature rules in one table
    const { nq, nqr, nqx } = this.quad;
    const gl = leggauss(nq);
    const glr = leggauss(nqr);
    const glx = leggauss(nqx);
    const tab = new Float32Array(npols * npols + 2 * nq + 2 * nqr + 2 * nqx);
    tab.set(surf.umat, 0);
    let o = npols * npols;
    tab.set(gl.x, o);
    tab.set(gl.w, o + nq);
    o += 2 * nq;
    for (let i = 0; i < nqr; i++) {
      tab[o + i] = 0.5 * (glr.x[i] + 1);
      tab[o + nqr + i] = 0.5 * glr.w[i];
    }
    o += 2 * nqr;
    tab.set(glx.x, o);
    tab.set(glx.w, o + nqx);
    this.tables = gpu.storage(tab);
    // oversampled sources for the far field
    const ov = surf.oversampled(this.nover);
    const N = ov.w.length;
    const sg = new Float32Array(8 * N);
    for (let i = 0; i < N; i++) {
      sg[8 * i] = ov.x[i];
      sg[8 * i + 1] = ov.x[N + i];
      sg[8 * i + 2] = ov.x[2 * N + i];
      sg[8 * i + 3] = ov.w[i];
      sg[8 * i + 4] = ov.n[i];
      sg[8 * i + 5] = ov.n[N + i];
      sg[8 * i + 6] = ov.n[2 * N + i];
    }
    this.srcgeo = gpu.storage(sg);
    this.srcq = gpu.storage(16 * N);
    this.qbuf = gpu.storage(2 * 8 * npts);
    this.out = gpu.storage(6 * 8 * npts);

    // self-patch polar segments for every node, from the first fundamental form at the node
    this.selfItems = new Array(npts);
    for (let i = 0; i < npts; i++) this.selfItems[i] = this.selfSegments(i);

    const xs = surf.srcvals.subarray(0, 3 * npts);
    const nrm = surf.srcvals.subarray(9 * npts, 12 * npts);
    this.wall = await this.makeTargets(xs, nrm, true, 'near quadrature');
    await gpu.sync();
    const setupSeconds = now() - t0;
    this.info = {
      npts,
      noversampled: N,
      npairs: this.wall.pairs.npairs,
      nleaves: this.wall.nleaves,
      weightsBytes: this.wall.W.size,
      setupSeconds,
    };
    return this.info;
  }

  destroy() {
    this.destroyed = true;
    for (const b of [this.gpar, this.tables, this.srcgeo, this.srcq, this.qbuf, this.out, this.uniformLeaves, this.uniformWeights, this.uniformDense, this.uniformSparse]) b?.destroy();
    if (this.wall) this.freeTargets(this.wall);
  }

  /** Bytes of near weights for npairs pairs. */
  weightsBytes(npairs: number): number {
    return npairs * 6 * this.npols * 8;
  }

  /** Estimated bytes of near weights for the surface nodes, before setup. */
  estimateWeightsBytes(): number {
    const { npts } = this;
    const xs = this.surf.srcvals.subarray(0, 3 * npts);
    let cnt = 0;
    const { npatches } = this.surf;
    const cen = this.surf.centroids;
    // sample every 16th node
    for (let i = 0; i < npts; i += 16) {
      for (let j = 0; j < npatches; j++) {
        const d2 = (xs[i] - cen[j]) ** 2 + (xs[npts + i] - cen[npatches + j]) ** 2 + (xs[2 * npts + i] - cen[2 * npatches + j]) ** 2;
        if (d2 <= this.r2[j]) cnt++;
      }
    }
    return this.weightsBytes(Math.ceil((cnt * npts) / Math.ceil(npts / 16)));
  }

  // ---------------------------------------------------------------- near pairs

  /** Pairs (target, patch) with |x_t - c_j|^2 <= r2_j, grouped by target, patches ascending. */
  nearPairs(x: Float64Array, onSurface: boolean): NearPairs {
    const n = x.length / 3;
    const { npatches, centroids: cen, patchId, uvNodes } = this.surf;
    const r2 = this.r2;
    const tList: number[] = [];
    const pList: number[] = [];
    const fList: number[] = [];
    const rowPtr = new Uint32Array(n + 1);
    const exact = this.surf.exact;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const yi = x[n + i];
      const zi = x[2 * n + i];
      for (let j = 0; j < npatches; j++) {
        const d2 = (xi - cen[j]) ** 2 + (yi - cen[npatches + j]) ** 2 + (zi - cen[2 * npatches + j]) ** 2;
        if (d2 > r2[j]) continue;
        tList.push(i);
        pList.push(j);
        if (!onSurface) {
          fList.push(0, 0, 0, 0, 0, 0, 0, 0);
        } else if (patchId[i] === j) {
          fList.push(uvNodes[i], uvNodes[n + i], 2, 0, 0, 0, 0, 0);
        } else if (exact) {
          // the target mapped into this patch's face chart: c_t = x_t / (x_t . n)
          const par = exact.params.subarray(13 * j, 13 * j + 13);
          const xn = xi * par[0] + yi * par[1] + zi * par[2];
          if (xn < 0.2) {
            fList.push(0, 0, 0, 0, 0, 0, 0, 0);
          } else {
            const cx = xi / xn;
            const cy = yi / xn;
            const cz = zi / xn;
            const st = cx * par[3] + cy * par[4] + cz * par[5];
            const tt = cx * par[6] + cy * par[7] + cz * par[8];
            fList.push((2 * (st - par[9])) / (par[10] - par[9]) - 1, (2 * (tt - par[11])) / (par[12] - par[11]) - 1, 1, 0, 0, 0, 0, 0);
          }
        } else {
          const pre = this.preimage(j, xi, yi, zi);
          if (pre) fList.push(pre[0], pre[1], 1, 0, pre[2], pre[3], pre[4], 0);
          else fList.push(0, 0, 0, 0, 0, 0, 0, 0);
        }
      }
      rowPtr[i + 1] = tList.length;
    }
    return { pairT: Int32Array.from(tList), pairP: Int32Array.from(pList), rowPtr, pairF: Float32Array.from(fList), npairs: tList.length };
  }

  /**
   * For the interpolated geometry: the preimage (u, v) of the point x in the Legendre
   * interpolant of patch j, extended beyond [-1, 1]^2, by Gauss-Newton from the nearest
   * node, and the gap e = x(u, v) - x. Returns [u, v, e_x, e_y, e_z], or null when the
   * point does not map close to the patch (then a plain difference is used).
   */
  private preimage(j: number, x: number, y: number, z: number): number[] | null {
    const { npols, npts, norder } = this;
    const n1 = norder + 1;
    const sc = this.surf.srccoefs;
    const sv = this.surf.srcvals;
    const base = j * npols;
    let best = 0;
    let bd = Infinity;
    for (let l = 0; l < npols; l++) {
      const d = (sv[base + l] - x) ** 2 + (sv[npts + base + l] - y) ** 2 + (sv[2 * npts + base + l] - z) ** 2;
      if (d < bd) {
        bd = d;
        best = l;
      }
    }
    let u = this.surf.uvNodes[base + best];
    let v = this.surf.uvNodes[npts + base + best];
    const Pu = new Float64Array(n1);
    const Pv = new Float64Array(n1);
    const Du = new Float64Array(n1);
    const Dv = new Float64Array(n1);
    const leg = (t: number, P: Float64Array, Dd: Float64Array) => {
      P[0] = 1;
      Dd[0] = 0;
      if (norder >= 1) {
        P[1] = t;
        Dd[1] = 1;
      }
      for (let m = 1; m < norder; m++) {
        P[m + 1] = ((2 * m + 1) * t * P[m] - m * P[m - 1]) / (m + 1);
        Dd[m + 1] = Dd[m - 1] + (2 * m + 1) * P[m];
      }
    };
    const g = new Float64Array(9); // x, x_u, x_v of the xyz interpolant
    const evalAt = () => {
      leg(u, Pu, Du);
      leg(v, Pv, Dv);
      g.fill(0);
      for (let c = 0; c < 3; c++) {
        for (let b = 0; b < n1; b++) {
          for (let a = 0; a < n1; a++) {
            const cf = sc[c * npts + base + b * n1 + a];
            g[c] += cf * Pu[a] * Pv[b];
            g[3 + c] += cf * Du[a] * Pv[b];
            g[6 + c] += cf * Pu[a] * Dv[b];
          }
        }
      }
    };
    for (let it = 0; it < 30; it++) {
      evalAt();
      const r = [x - g[0], y - g[1], z - g[2]];
      const a11 = g[3] ** 2 + g[4] ** 2 + g[5] ** 2;
      const a12 = g[3] * g[6] + g[4] * g[7] + g[5] * g[8];
      const a22 = g[6] ** 2 + g[7] ** 2 + g[8] ** 2;
      const b1 = g[3] * r[0] + g[4] * r[1] + g[5] * r[2];
      const b2 = g[6] * r[0] + g[7] * r[1] + g[8] * r[2];
      const det = a11 * a22 - a12 * a12;
      if (!(det > 0)) return null;
      const du = (a22 * b1 - a12 * b2) / det;
      const dv = (a11 * b2 - a12 * b1) / det;
      u += du;
      v += dv;
      if (Math.abs(u) > 3 || Math.abs(v) > 3) return null;
      if (Math.abs(du) + Math.abs(dv) < 1e-13) break;
    }
    if (Math.abs(u) > 2 || Math.abs(v) > 2) return null;
    evalAt();
    const e = [g[0] - x, g[1] - y, g[2] - z];
    if (Math.hypot(e[0], e[1], e[2]) > 0.1 * this.surf.rads[j]) return null;
    return [u, v, ...e];
  }

  /** The sinh-substituted segments along the four edges for the self patch of node i: 8 floats each. */
  private selfSegments(i: number): Float32Array {
    const { npts } = this;
    const sv = this.surf.srcvals;
    const u0 = this.surf.uvNodes[i];
    const v0 = this.surf.uvNodes[npts + i];
    const ru = [sv[3 * npts + i], sv[4 * npts + i], sv[5 * npts + i]];
    const rv = [sv[6 * npts + i], sv[7 * npts + i], sv[8 * npts + i]];
    const E = ru[0] ** 2 + ru[1] ** 2 + ru[2] ** 2;
    const F = ru[0] * rv[0] + ru[1] * rv[1] + ru[2] * rv[2];
    const G = rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2;
    const det = Math.sqrt(Math.max(E * G - F * F, 1e-300));
    const k = Math.abs(this.k);
    const { subLen, oscLen } = this.quad;
    const items: number[] = [];
    for (let e = 0; e < 4; e++) {
      const h = [1 - u0, -1 - u0, 1 - v0, -1 - v0][e];
      if (Math.abs(h) < 1e-12) continue;
      let x1: number, x2: number, xc: number, heff: number, arc: number;
      if (e < 2) {
        x1 = -1 - v0;
        x2 = 1 - v0;
        xc = (-h * F) / G;
        heff = (Math.abs(h) * det) / G;
        arc = Math.sqrt(G);
      } else {
        x1 = -1 - u0;
        x2 = 1 - u0;
        xc = (-h * F) / E;
        heff = (Math.abs(h) * det) / E;
        arc = Math.sqrt(E);
      }
      const xi0 = Math.asinh((x1 - xc) / heff);
      const xi1 = Math.asinh((x2 - xc) / heff);
      // sub-intervals of length <= subLen in xi whose extent along the edge is <= dxmax
      const dxmax = k * arc > 0 ? oscLen / (k * arc) : Infinity;
      let xi = xi0;
      let cnt = 0;
      while (xi < xi1 && cnt < 256) {
        const xh = xc + heff * Math.sinh(xi);
        let xn = Math.min(xi + subLen, Math.asinh((xh + dxmax - xc) / heff));
        if (xn > xi1 || cnt === 255) xn = xi1;
        items.push(1, e, xi, xn, h, xc, heff, 0);
        xi = xn;
        cnt++;
      }
    }
    return Float32Array.from(items);
  }

  // ------------------------------------------------------------ near weights

  /** Build the near pairs, quadtree leaves and near weights for a set of targets. */
  private async makeTargets(x: Float64Array, nrm: Float64Array | null, onSurface: boolean, stage: string): Promise<TargetSet> {
    const { gpu } = this;
    const n = x.length / 3;
    const tg = new Float32Array(8 * n);
    for (let i = 0; i < n; i++) {
      tg[8 * i] = x[i];
      tg[8 * i + 1] = x[n + i];
      tg[8 * i + 2] = x[2 * n + i];
      if (nrm) {
        tg[8 * i + 4] = nrm[i];
        tg[8 * i + 5] = nrm[n + i];
        tg[8 * i + 6] = nrm[2 * n + i];
      }
    }
    const pairs = this.nearPairs(x, onSurface);
    const wbytes = this.weightsBytes(pairs.npairs);
    const lim = gpu.device.limits;
    if (wbytes > lim.maxStorageBufferBindingSize || wbytes > lim.maxBufferSize) {
      throw new Error(
        `the near weights need ${(wbytes / 2 ** 20).toFixed(0)} MB but this GPU allows buffers of at most ` +
          `${(Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize) / 2 ** 20).toFixed(0)} MB; use fewer patches or a lower order`,
      );
    }
    const pi = new Int32Array(2 * Math.max(1, pairs.npairs));
    for (let p = 0; p < pairs.npairs; p++) {
      pi[2 * p] = pairs.pairT[p];
      pi[2 * p + 1] = pairs.pairP[p];
    }
    const set: TargetSet = {
      n,
      targ: gpu.storage(tg),
      pairs,
      pairsI: gpu.storage(pi),
      pairsF: gpu.storage(pairs.npairs ? pairs.pairF : new Float32Array(8)),
      rowPtr: gpu.storage(pairs.rowPtr),
      pairP: gpu.storage(pairs.npairs ? new Uint32Array(pairs.pairP.buffer, pairs.pairP.byteOffset, pairs.npairs) : new Uint32Array(1)),
      W: gpu.storage(Math.max(16, wbytes)),
      nleaves: 0,
    };
    if (pairs.npairs > 0) set.nleaves = await this.computeWeights(set, stage);
    return set;
  }

  private freeTargets(s: TargetSet) {
    for (const b of [s.targ, s.pairsI, s.pairsF, s.rowPtr, s.pairP, s.W]) b.destroy();
  }

  /** Quadtree leaves and near weights for all pairs of a target set, in paced chunks. Returns the leaf count. */
  private async computeWeights(set: TargetSet, stage: string): Promise<number> {
    const { gpu, quad, leaf } = this;
    const { npairs, pairF } = set.pairs;
    const leavesPipe = await gpu.pipeline('leaves', BUILD_LEAVES, 'build_leaves');
    const weightsPipe = await gpu.pipeline('weights', NEAR_WEIGHTS, 'near_weights');
    const pacer = new Pacer(this.pace, 512);
    let nleaves = 0;
    let p0 = 0;
    let itemsPerPair = 40;
    const maxItemsBytes = 48 * 2 ** 20;
    while (p0 < npairs) {
      if (this.destroyed) throw new Error('operator destroyed');
      const cap = Math.min(65535, npairs - p0, Math.max(256, Math.floor(maxItemsBytes / (32 * itemsPerPair))));
      const n = pacer.next(cap);
      const t0 = now();
      // pass 1: leaf counts
      const counts = gpu.storage(4 * n);
      const ex = this.surf.exact;
      const ga = ex ? ex.a : 0;
      const gpe = ex ? ex.p : 0;
      const gmode = ex ? 0 : 1;
      const uw = new UniformWriter(11);
      uw.u32(n).u32(p0).u32(0).i32(leaf.nlmax).f32(leaf.rfacQ).f32(Math.abs(this.k)).f32(leaf.kRMax).f32(ga).i32(gpe).i32(gmode).i32(this.norder);
      gpu.write(this.uniformLeaves, new Uint8Array(uw.bytes));
      const dummyItems = gpu.storage(16);
      const dummyOffsets = gpu.storage(16);
      gpu.dispatch(leavesPipe, gpu.bind(leavesPipe, [this.uniformLeaves, set.pairsI, set.pairsF, set.targ, this.gpar, counts, dummyOffsets, dummyItems]), Math.ceil(n / 64));
      const cnt = new Uint32Array(await gpu.read(counts, 4 * n));
      // self pairs: the CPU segments
      const ptr = new Uint32Array(n + 1);
      for (let q = 0; q < n; q++) {
        const p = p0 + q;
        const c = pairF[8 * p + 2] === 2 ? this.selfItems[set.pairs.pairT[p]].length / 8 : cnt[q];
        if (pairF[8 * p + 2] !== 2) nleaves += cnt[q];
        ptr[q + 1] = ptr[q] + c;
      }
      const nitems = ptr[n];
      const items = new Float32Array(8 * Math.max(1, nitems));
      for (let q = 0; q < n; q++) {
        const p = p0 + q;
        if (pairF[8 * p + 2] === 2) items.set(this.selfItems[set.pairs.pairT[p]], 8 * ptr[q]);
      }
      const itemsBuf = gpu.storage(items);
      const ptrBuf = gpu.storage(ptr);
      // pass 2: fill the leaves
      const uw2 = new UniformWriter(11);
      uw2.u32(n).u32(p0).u32(1).i32(leaf.nlmax).f32(leaf.rfacQ).f32(Math.abs(this.k)).f32(leaf.kRMax).f32(ga).i32(gpe).i32(gmode).i32(this.norder);
      gpu.write(this.uniformLeaves, new Uint8Array(uw2.bytes));
      gpu.dispatch(leavesPipe, gpu.bind(leavesPipe, [this.uniformLeaves, set.pairsI, set.pairsF, set.targ, this.gpar, counts, ptrBuf, itemsBuf]), Math.ceil(n / 64));
      // the weights
      const uw3 = new UniformWriter(11);
      uw3.u32(n).u32(p0).i32(this.norder).i32(this.npols).i32(quad.nq).i32(quad.nqr).i32(quad.nqx).f32(this.k).f32(ga).i32(gpe).i32(gmode);
      gpu.write(this.uniformWeights, new Uint8Array(uw3.bytes));
      gpu.dispatch(weightsPipe, gpu.bind(weightsPipe, [this.uniformWeights, set.pairsI, set.pairsF, set.targ, this.gpar, ptrBuf, itemsBuf, this.tables, set.W]), n);
      await gpu.sync();
      const dt = now() - t0;
      pacer.record(n, dt);
      itemsPerPair = Math.max(1, nitems / n);
      for (const b of [counts, dummyItems, dummyOffsets, itemsBuf, ptrBuf]) b.destroy();
      p0 += n;
      this.progress(stage, p0 / npairs);
    }
    return nleaves;
  }

  // ------------------------------------------------------------- application

  /** Values of q at the oversampled nodes, per patch: q_o = M q. */
  private oversample(q: CVec, out: Float32Array, offset: number) {
    const { npols, npolsO, overMat } = this;
    const np = this.surf.npatches;
    for (let j = 0; j < np; j++) {
      for (let kk = 0; kk < npolsO; kk++) {
        let sr = 0;
        let si = 0;
        for (let m = 0; m < npols; m++) {
          const M = overMat[kk * npols + m];
          sr += M * q[2 * (j * npols + m)];
          si += M * q[2 * (j * npols + m) + 1];
        }
        const idx = 4 * (j * npolsO + kk) + offset;
        out[idx] = sr;
        out[idx + 1] = si;
      }
    }
  }

  /** Far and near parts of the selected kernels at a target set. sel: 6 entries in {0, 1, 2}; reqs: [kernel, density] pairs. */
  private async applyAt(set: TargetSet, qa: CVec, qb: CVec | null, sel: number[], reqs: [number, number][]): Promise<Float32Array> {
    const { gpu, npts, npolsO } = this;
    const N = npolsO * this.surf.npatches;
    const sq = new Float32Array(4 * N);
    this.oversample(qa, sq, 0);
    if (qb) this.oversample(qb, sq, 2);
    gpu.write(this.srcq, sq);
    const qd = new Float32Array(2 * 2 * npts);
    for (let i = 0; i < 2 * npts; i++) qd[i] = qa[i];
    if (qb) for (let i = 0; i < 2 * npts; i++) qd[2 * npts + i] = qb[i];
    gpu.write(this.qbuf, qd);
    const out = set.n === this.wall.n ? this.out : gpu.storage(6 * 8 * set.n);
    gpu.clear(out);
    const selmask = sel.reduce((acc, s, m) => acc | (s << (4 * m)), 0);
    const densePipe = await gpu.pipeline('dense', DENSE_APPLY, 'dense_apply');
    const group = gpu.bind(densePipe, [this.uniformDense, set.targ, this.srcgeo, this.srcq, set.rowPtr, set.pairP, out]);
    const pacer = new Pacer(this.pace, Math.max(1, Math.floor(4e8 / (set.n * npolsO * 60))));
    const np = this.surf.npatches;
    let j0 = 0;
    while (j0 < np) {
      const n = pacer.next(np - j0);
      const t0 = now();
      const uw = new UniformWriter(6);
      uw.u32(set.n).u32(j0).u32(j0 + n).u32(npolsO).f32(this.k).u32(selmask);
      gpu.write(this.uniformDense, new Uint8Array(uw.bytes));
      gpu.dispatch(densePipe, group, Math.ceil(set.n / DENSE_WG));
      await gpu.sync();
      pacer.record(n, now() - t0);
      j0 += n;
    }
    if (set.pairs.npairs > 0 && reqs.length > 0) {
      const sparsePipe = await gpu.pipeline('sparse', SPARSE_APPLY, 'sparse_apply');
      const uw = new UniformWriter(8);
      uw.u32(set.n).u32(this.npols).u32(npts).u32(reqs.length);
      for (let r = 0; r < 4; r++) uw.u32(r < reqs.length ? reqs[r][0] | (reqs[r][1] << 8) : 0);
      gpu.write(this.uniformSparse, new Uint8Array(uw.bytes));
      gpu.dispatch(sparsePipe, gpu.bind(sparsePipe, [this.uniformSparse, set.rowPtr, set.pairP, set.W, this.qbuf, out]), Math.ceil(set.n / 128));
    }
    const res = new Float32Array(await gpu.read(out, 6 * 8 * set.n));
    if (out !== this.out) out.destroy();
    return res;
  }

  zid(): [number, number] {
    return [(-1) ** this.ifinout / 2, -this.alpha / 4];
  }

  /** The operator without the identity term z. */
  async applyK(sigma: CVec): Promise<CVec> {
    const { npts, k, alpha: a, lam } = this;
    const F = await this.applyAt(this.wall, sigma, null, [1, 1, 1, 0, 1, 0], [[0, 0], [1, 0], [2, 0], [4, 0]]);
    const phi1 = new Float64Array(2 * npts);
    const phi2 = new Float64Array(2 * npts);
    const res = new Float64Array(2 * npts);
    for (let i = 0; i < npts; i++) {
      phi1[2 * i] = F[2 * (npts + i)];
      phi1[2 * i + 1] = F[2 * (npts + i) + 1];
      phi2[2 * i] = F[2 * (2 * npts + i)];
      phi2[2 * i + 1] = F[2 * (2 * npts + i) + 1];
      const s4r = F[2 * (4 * npts + i)];
      const s4i = F[2 * (4 * npts + i) + 1];
      // res = F0 + i k lam F4
      res[2 * i] = F[2 * i] - k * lam * s4i;
      res[2 * i + 1] = F[2 * i + 1] + k * lam * s4r;
    }
    const G = await this.applyAt(this.wall, phi1, phi2, [0, 0, 2, 1, 0, 1], [[2, 1], [3, 0], [5, 0]]);
    const sgn = (-1) ** this.ifinout;
    for (let i = 0; i < npts; i++) {
      const g2r = G[2 * (2 * npts + i)];
      const g2i = G[2 * (2 * npts + i) + 1];
      const g3r = G[2 * (3 * npts + i)];
      const g3i = G[2 * (3 * npts + i) + 1];
      const g5r = G[2 * (5 * npts + i)];
      const g5i = G[2 * (5 * npts + i) + 1];
      // res += i a (G2 + G3) - k a lam G5 + sgn k a lam phi1 / 2
      res[2 * i] += -a * (g2i + g3i) - k * a * lam * g5r + (sgn * k * a * lam * phi1[2 * i]) / 2;
      res[2 * i + 1] += a * (g2r + g3r) - k * a * lam * g5i + (sgn * k * a * lam * phi1[2 * i + 1]) / 2;
    }
    return res;
  }

  /** phi1 = S_kappa[sigma] on the surface. */
  async sKappa(sigma: CVec): Promise<CVec> {
    const F = await this.applyAt(this.wall, sigma, null, [0, 1, 0, 0, 0, 0], [[1, 0]]);
    const { npts } = this;
    const phi1 = new Float64Array(2 * npts);
    for (let i = 0; i < 2 * npts; i++) phi1[i] = F[2 * npts + i];
    return phi1;
  }

  /** One of the six operators applied to q on the surface (principal value), for tests. */
  async applySingle(m: number, q: CVec): Promise<CVec> {
    const sel = [0, 0, 0, 0, 0, 0];
    sel[m] = 1;
    const F = await this.applyAt(this.wall, q, null, sel, [[m, 0]]);
    const { npts } = this;
    const out = new Float64Array(2 * npts);
    for (let i = 0; i < 2 * npts; i++) out[i] = F[2 * m * npts + i];
    return out;
  }

  async solve(rhs: CVec, epsGmres = 1e-5, numit = 200, onIter?: (it: number, rres: number) => void): Promise<GmresResult & { phi1: CVec; seconds: number }> {
    const t0 = now();
    const r = await gmres((x) => this.applyK(x), rhs, this.surf.wts, this.zid(), epsGmres, numit, onIter);
    const phi1 = await this.sKappa(r.x);
    return { ...r, phi1, seconds: now() - t0 };
  }

  /** u = S_k sigma + i alpha D_k phi1 at the surface nodes, as the limit from the side selected by ifinout. */
  async evalOnSurface(sigma: CVec, phi1: CVec): Promise<CVec> {
    const F = await this.applyAt(this.wall, sigma, phi1, [0, 0, 0, 0, 1, 2], [[4, 0], [5, 1]]);
    return this.combine(F, this.npts, phi1);
  }

  /** u = S_k sigma + i alpha D_k phi1 at off-surface targets x (3, n), in chunks. */
  async eval(x: Float64Array, sigma: CVec, phi1: CVec, stage = 'field'): Promise<CVec> {
    const n = x.length / 3;
    const u = new Float64Array(2 * n);
    const chunk = 4096;
    for (let i0 = 0; i0 < n; i0 += chunk) {
      const m = Math.min(chunk, n - i0);
      const xs = new Float64Array(3 * m);
      for (let d = 0; d < 3; d++) xs.set(x.subarray(d * n + i0, d * n + i0 + m), d * m);
      const set = await this.makeTargets(xs, null, false, stage);
      const F = await this.applyAt(set, sigma, phi1, [0, 0, 0, 0, 1, 2], [[4, 0], [5, 1]]);
      this.freeTargets(set);
      u.set(this.combine(F, m, null), 2 * i0);
      this.progress(stage, (i0 + m) / n);
    }
    return u;
  }

  private combine(F: Float32Array, n: number, phi1OnSurface: CVec | null): CVec {
    const a = this.alpha;
    const u = new Float64Array(2 * n);
    const jump = phi1OnSurface ? (a * (-1) ** (this.ifinout + 1)) / 2 : 0;
    for (let i = 0; i < n; i++) {
      const s4r = F[2 * (4 * n + i)];
      const s4i = F[2 * (4 * n + i) + 1];
      const d5r = F[2 * (5 * n + i)];
      const d5i = F[2 * (5 * n + i) + 1];
      // u = S + i a D (+ i a (-1)^(ifinout+1) phi1 / 2 on the surface)
      u[2 * i] = s4r - a * d5i;
      u[2 * i + 1] = s4i + a * d5r;
      if (phi1OnSurface) {
        u[2 * i] += -jump * phi1OnSurface[2 * i + 1];
        u[2 * i + 1] += jump * phi1OnSurface[2 * i];
      }
    }
    return u;
  }

  /** The near weights of the surface nodes, for tests: (npairs, 6, npols) complex interleaved. */
  async readWallWeights(): Promise<{ pairs: NearPairs; W: Float32Array }> {
    const W = new Float32Array(await this.gpu.read(this.wall.W, this.weightsBytes(this.wall.pairs.npairs)));
    return { pairs: this.wall.pairs, W };
  }
}

export { NT };
