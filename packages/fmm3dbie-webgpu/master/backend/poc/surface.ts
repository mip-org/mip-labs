/**
 * Quadrilateral patch surfaces in the fmm3dbie convention (iptype 11), and the
 * rounded cube (superellipsoid) built from them. A direct port of
 * fmm3dbie_gpu/surface.py from the CUDA proof of concept.
 *
 * Each patch is the image of the reference square [-1, 1]^2 and carries
 * (norder+1)^2 tensor Gauss-Legendre nodes, with the u index varying fastest,
 * and an unnormalized Legendre basis P_j(u) P_i(v) in the same order. Arrays
 * with a shape (r, c) are stored row-major in one Float64Array, index r*c + i.
 */
import { leggauss, legendreVander } from './gauss.ts';

/** Reference nodes (2, npols) with u in the first row, and the tensor weights. */
export function discNodes(norder: number): { uv: Float64Array; w: Float64Array } {
  const n1 = norder + 1;
  const { x, w } = leggauss(n1);
  const npols = n1 * n1;
  const uv = new Float64Array(2 * npols);
  const ww = new Float64Array(npols);
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n1; j++) {
      const n = i * n1 + j;
      uv[n] = x[j];
      uv[npols + n] = x[i];
      ww[n] = w[i] * w[j];
    }
  }
  return { uv, w: ww };
}

/** Basis values (npols, npts): index n = i*(norder+1) + j is P_j(u) P_i(v). */
export function basis2d(uv: Float64Array, norder: number): Float64Array {
  const npts = uv.length / 2;
  const n1 = norder + 1;
  const Pu = legendreVander(uv.subarray(0, npts), norder);
  const Pv = legendreVander(uv.subarray(npts), norder);
  const B = new Float64Array(n1 * n1 * npts);
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n1; j++) {
      const n = i * n1 + j;
      for (let q = 0; q < npts; q++) B[n * npts + q] = Pv[i * npts + q] * Pu[j * npts + q];
    }
  }
  return B;
}

/** U (npols, npols) with coefs = U @ vals for values at the tensor GL nodes. */
export function valsToCoefsMatrix(norder: number): Float64Array {
  const n1 = norder + 1;
  const { x, w } = leggauss(n1);
  const P = legendreVander(x, norder);
  const U1 = new Float64Array(n1 * n1);
  for (let k = 0; k < n1; k++) for (let j = 0; j < n1; j++) U1[k * n1 + j] = ((2 * k + 1) / 2) * P[k * n1 + j] * w[j];
  const npols = n1 * n1;
  const U = new Float64Array(npols * npols);
  for (let i = 0; i < n1; i++) {
    for (let ii = 0; ii < n1; ii++) {
      for (let j = 0; j < n1; j++) {
        for (let jj = 0; jj < n1; jj++) U[(i * n1 + ii) * npols + (j * n1 + jj)] = U1[i * n1 + j] * U1[ii * n1 + jj];
      }
    }
  }
  return U;
}

/** M (npols_o, npols) with values at the order-nover nodes = M @ values at the order-norder nodes. */
export function oversampleMatrix(norder: number, nover: number): Float64Array {
  const { uv } = discNodes(nover);
  const npolsO = (nover + 1) ** 2;
  const npols = (norder + 1) ** 2;
  const B = basis2d(uv, norder); // (npols, npols_o)
  const U = valsToCoefsMatrix(norder);
  const M = new Float64Array(npolsO * npols);
  for (let k = 0; k < npolsO; k++) {
    for (let m = 0; m < npols; m++) {
      let s = 0;
      for (let n = 0; n < npols; n++) s += B[n * npolsO + k] * U[n * npols + m];
      M[k * npols + m] = s;
    }
  }
  return M;
}

/** (n, e1, e2) with e1 x e2 = n for the six faces of the cube. */
export const FACES: [number[], number[], number[]][] = [
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
  [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
  [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
];

/** m+1 breakpoints on [-1, 1], blending uniform and sine-clustered spacing. */
export function gradedBreaks(m: number, grading: number): Float64Array {
  const br = new Float64Array(m + 1);
  for (let i = 0; i <= m; i++) {
    const t = -1 + (2 * i) / m;
    br[i] = (1 - grading) * t + grading * Math.sin(0.5 * Math.PI * t);
  }
  return br;
}

/**
 * xyz, r_u, r_v (9, n) of the face chart par = (n, e1, e2, s0, s1, t0, t1) of the
 * superellipsoid |x|^p = a^p at reference points uv (2, n).
 */
export function superellipsoidChart(par: ArrayLike<number>, a: number, p: number, uv: Float64Array): Float64Array {
  const n = uv.length / 2;
  const out = new Float64Array(9 * n);
  const hs = 0.5 * (par[10] - par[9]);
  const ht = 0.5 * (par[12] - par[11]);
  const c = [0, 0, 0];
  const gN = [0, 0, 0];
  for (let q = 0; q < n; q++) {
    const s = par[9] + (uv[q] + 1) * hs;
    const t = par[11] + (uv[n + q] + 1) * ht;
    let S = 0;
    for (let d = 0; d < 3; d++) {
      c[d] = par[d] + s * par[3 + d] + t * par[6 + d];
      S += Math.pow(c[d], p);
    }
    const N = Math.pow(S, 1 / p);
    let dNu = 0;
    let dNv = 0;
    for (let d = 0; d < 3; d++) {
      gN[d] = Math.pow(c[d] / N, p - 1);
      dNu += gN[d] * par[3 + d];
      dNv += gN[d] * par[6 + d];
    }
    for (let d = 0; d < 3; d++) {
      out[d * n + q] = (a * c[d]) / N;
      out[(3 + d) * n + q] = a * hs * (par[3 + d] / N - (c[d] * dNu) / (N * N));
      out[(6 + d) * n + q] = a * ht * (par[6 + d] / N - (c[d] * dNv) / (N * N));
    }
  }
  return out;
}

export interface ExactGeometry {
  params: Float64Array; // (npatches, 13)
  a: number;
  p: number;
}

export class Surface {
  readonly npols: number;
  readonly npts: number;
  readonly npatches: number;
  readonly uvs: Float64Array; // (2, npols) reference nodes
  readonly wref: Float64Array;
  readonly umat: Float64Array; // (npols, npols)
  readonly srccoefs: Float64Array; // (9, npts)
  readonly jac: Float64Array;
  readonly wts: Float64Array;
  readonly centroids: Float64Array; // (3, npatches)
  readonly rads: Float64Array;
  readonly patchId: Int32Array;
  readonly uvNodes: Float64Array; // (2, npts)
  readonly norder: number;
  readonly srcvals: Float64Array; // (12, npts): xyz, r_u, r_v, unit normal
  /** the analytic chart of every patch, or null for the interpolated geometry */
  readonly exact: ExactGeometry | null;

  constructor(norder: number, srcvals: Float64Array, exact: ExactGeometry | null = null) {
    this.norder = norder;
    this.srcvals = srcvals;
    this.exact = exact;
    this.npols = (norder + 1) ** 2;
    this.npts = srcvals.length / 12;
    if (this.npts % this.npols !== 0) throw new Error('srcvals is not a whole number of patches');
    this.npatches = this.npts / this.npols;
    const { uv, w } = discNodes(norder);
    this.uvs = uv;
    this.wref = w;
    this.umat = valsToCoefsMatrix(norder);
    const { npols, npts, npatches } = this;
    this.srccoefs = new Float64Array(9 * npts);
    for (let c = 0; c < 9; c++) {
      for (let pch = 0; pch < npatches; pch++) {
        const base = pch * npols;
        for (let n = 0; n < npols; n++) {
          let s = 0;
          for (let j = 0; j < npols; j++) s += this.umat[n * npols + j] * srcvals[c * npts + base + j];
          this.srccoefs[c * npts + base + n] = s;
        }
      }
    }
    this.jac = new Float64Array(npts);
    this.wts = new Float64Array(npts);
    for (let i = 0; i < npts; i++) {
      const ru = [srcvals[3 * npts + i], srcvals[4 * npts + i], srcvals[5 * npts + i]];
      const rv = [srcvals[6 * npts + i], srcvals[7 * npts + i], srcvals[8 * npts + i]];
      const cx = ru[1] * rv[2] - ru[2] * rv[1];
      const cy = ru[2] * rv[0] - ru[0] * rv[2];
      const cz = ru[0] * rv[1] - ru[1] * rv[0];
      this.jac[i] = Math.hypot(cx, cy, cz);
      this.wts[i] = this.jac[i] * w[i % npols];
    }
    // centroid and radius of each patch from a 5x5 sample grid of its chart
    const g5 = new Float64Array(50);
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        g5[i * 5 + j] = -1 + 0.5 * i;
        g5[25 + i * 5 + j] = -1 + 0.5 * j;
      }
    }
    this.centroids = new Float64Array(3 * npatches);
    this.rads = new Float64Array(npatches);
    for (let pch = 0; pch < npatches; pch++) {
      const g = this.evalGeometry(pch, g5);
      const cen = [0, 0, 0];
      for (let k = 0; k < 25; k++) for (let d = 0; d < 3; d++) cen[d] += g[d * 25 + k] / 25;
      let r2 = 0;
      for (let k = 0; k < 25; k++) {
        let s = 0;
        for (let d = 0; d < 3; d++) s += (g[d * 25 + k] - cen[d]) ** 2;
        r2 = Math.max(r2, s);
      }
      for (let d = 0; d < 3; d++) this.centroids[d * npatches + pch] = cen[d];
      this.rads[pch] = Math.sqrt(r2);
    }
    this.patchId = new Int32Array(npts);
    this.uvNodes = new Float64Array(2 * npts);
    for (let i = 0; i < npts; i++) {
      this.patchId[i] = Math.floor(i / npols);
      this.uvNodes[i] = uv[i % npols];
      this.uvNodes[npts + i] = uv[npols + (i % npols)];
    }
  }

  /**
   * xyz, r_u, r_v (9, n) of one patch at reference points uv (2, n): the exact chart
   * when there is one, otherwise the Legendre interpolant of the node data, as in
   * fmm3dbie.
   */
  evalGeometry(patch: number, uv: Float64Array): Float64Array {
    const e = this.exact;
    if (e) return superellipsoidChart(e.params.subarray(13 * patch, 13 * patch + 13), e.a, e.p, uv);
    const n = uv.length / 2;
    const { npols, npts } = this;
    const B = basis2d(uv, this.norder);
    const out = new Float64Array(9 * n);
    const base = patch * npols;
    for (let c = 0; c < 9; c++) {
      for (let m = 0; m < npols; m++) {
        const cm = this.srccoefs[c * npts + base + m];
        for (let q = 0; q < n; q++) out[c * n + q] += cm * B[m * n + q];
      }
    }
    return out;
  }

  /** Nodes of the order-nover rule on every patch: xyz (3, N), unit normal (3, N), weights (N). */
  oversampled(nover: number): { x: Float64Array; n: Float64Array; w: Float64Array } {
    const { uv, w: wo } = discNodes(nover);
    const npolsO = (nover + 1) ** 2;
    const N = npolsO * this.npatches;
    const x = new Float64Array(3 * N);
    const nrm = new Float64Array(3 * N);
    const w = new Float64Array(N);
    for (let pch = 0; pch < this.npatches; pch++) {
      const g = this.evalGeometry(pch, uv);
      for (let k = 0; k < npolsO; k++) {
        const i = pch * npolsO + k;
        const ru = [g[3 * npolsO + k], g[4 * npolsO + k], g[5 * npolsO + k]];
        const rv = [g[6 * npolsO + k], g[7 * npolsO + k], g[8 * npolsO + k]];
        const cx = ru[1] * rv[2] - ru[2] * rv[1];
        const cy = ru[2] * rv[0] - ru[0] * rv[2];
        const cz = ru[0] * rv[1] - ru[1] * rv[0];
        const jac = Math.hypot(cx, cy, cz);
        x[i] = g[k];
        x[N + i] = g[npolsO + k];
        x[2 * N + i] = g[2 * npolsO + k];
        nrm[i] = cx / jac;
        nrm[N + i] = cy / jac;
        nrm[2 * N + i] = cz / jac;
        w[i] = jac * wo[k];
      }
    }
    return { x, n: nrm, w };
  }
}

/**
 * The superellipsoid |x|^p + |y|^p + |z|^p = a^p with m x m patches per cube
 * face, graded toward the face edges where the curvature is concentrated.
 */
export function roundedCube(a: number, p: number, m: number, norder: number, grading = 0.5): Surface {
  const { uv } = discNodes(norder);
  const npols = (norder + 1) ** 2;
  const br = gradedBreaks(m, grading);
  const npatches = 6 * m * m;
  const npts = npatches * npols;
  const params = new Float64Array(13 * npatches);
  const srcvals = new Float64Array(12 * npts);
  let pch = 0;
  for (const [n, e1, e2] of FACES) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        const par = [...n, ...e1, ...e2, br[i], br[i + 1], br[j], br[j + 1]];
        params.set(par, 13 * pch);
        const g = superellipsoidChart(par, a, p, uv);
        for (let k = 0; k < npols; k++) {
          const idx = pch * npols + k;
          for (let c = 0; c < 9; c++) srcvals[c * npts + idx] = g[c * npols + k];
          const ru = [g[3 * npols + k], g[4 * npols + k], g[5 * npols + k]];
          const rv = [g[6 * npols + k], g[7 * npols + k], g[8 * npols + k]];
          const cx = ru[1] * rv[2] - ru[2] * rv[1];
          const cy = ru[2] * rv[0] - ru[0] * rv[2];
          const cz = ru[0] * rv[1] - ru[1] * rv[0];
          const len = Math.hypot(cx, cy, cz);
          srcvals[9 * npts + idx] = cx / len;
          srcvals[10 * npts + idx] = cy / len;
          srcvals[11 * npts + idx] = cz / len;
        }
        pch++;
      }
    }
  }
  return new Surface(norder, srcvals, { params, a, p });
}

/** Exact volume of |x|^p + |y|^p + |z|^p <= a^p. */
export function superellipsoidVolume(a: number, p: number): number {
  return (8 * a ** 3 * gamma(1 + 1 / p) ** 3) / gamma(1 + 3 / p);
}

/** Lanczos approximation of the Gamma function (used only for the volume check). */
function gamma(z: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}
