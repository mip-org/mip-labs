/**
 * The showcase problem: a point source at x0 inside a rounded cube (the
 * superellipsoid |x|^p + |y|^p + |z|^p = a^p) with impedance walls. With time
 * dependence exp(-i omega t) the pressure amplitude u satisfies
 *
 *     (Delta + k^2) u = -delta(x - x0)   in the room,
 *     du/dn + i k lambda u = 0            on the walls,
 *
 * with n the outward normal and lambda = -beta, where beta > 0 is the wall
 * admittance. We write u = G + v with G the free-space field of the source and
 * find v from fmm3dbie's right-preconditioned combined field equation.
 *
 * Complex vectors are interleaved Float64Arrays: [re0, im0, re1, im1, ...].
 */

export type CVec = Float64Array;

export const ALPHA = 1.0;

/** Free-space Green's function exp(ikr)/(4 pi r) at points x (3, n) for a source x0. */
export function green(x: Float64Array, x0: ArrayLike<number>, k: number): CVec {
  const n = x.length / 3;
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const r = Math.hypot(x[i] - x0[0], x[n + i] - x0[1], x[2 * n + i] - x0[2]);
    const s = 1 / (4 * Math.PI * r);
    out[2 * i] = s * Math.cos(k * r);
    out[2 * i + 1] = s * Math.sin(k * r);
  }
  return out;
}

/** Normal derivative of the Green's function at x with unit normals nrm (3, n). */
export function greenDn(x: Float64Array, nrm: Float64Array, x0: ArrayLike<number>, k: number): CVec {
  const n = x.length / 3;
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const dx = x[i] - x0[0];
    const dy = x[n + i] - x0[1];
    const dz = x[2 * n + i] - x0[2];
    const r = Math.hypot(dx, dy, dz);
    const s = 1 / (4 * Math.PI * r);
    const gr = s * Math.cos(k * r);
    const gi = s * Math.sin(k * r);
    const dn = (dx * nrm[i] + dy * nrm[n + i] + dz * nrm[2 * n + i]) / r;
    // g * (ik - 1/r) * dn
    const fr = -1 / r;
    const fi = k;
    out[2 * i] = (gr * fr - gi * fi) * dn;
    out[2 * i + 1] = (gr * fi + gi * fr) * dn;
  }
  return out;
}

export function inside(x: ArrayLike<number>, a: number, p: number, margin = 0): boolean {
  return Math.pow(Math.abs(x[0] / a), p) + Math.pow(Math.abs(x[1] / a), p) + Math.pow(Math.abs(x[2] / a), p) < (1 - margin) ** p;
}

/** Impedance data for v = u - G: -(dG/dn + i k lambda G) at the wall nodes. */
export function roomRhs(xs: Float64Array, nrm: Float64Array, x0: ArrayLike<number>, k: number, lam: number): CVec {
  const g = green(xs, x0, k);
  const gn = greenDn(xs, nrm, x0, k);
  const n = g.length / 2;
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    // i k lam g = (-k lam g_im, k lam g_re)
    out[2 * i] = -(gn[2 * i] - k * lam * g[2 * i + 1]);
    out[2 * i + 1] = -(gn[2 * i + 1] + k * lam * g[2 * i]);
  }
  return out;
}

/** Impedance data of the field of a source outside the room (the self-test). */
export function selftestRhs(xs: Float64Array, nrm: Float64Array, xe: ArrayLike<number>, k: number, lam: number): CVec {
  const r = roomRhs(xs, nrm, xe, k, lam);
  for (let i = 0; i < r.length; i++) r[i] = -r[i];
  return r;
}

/** n deterministic pseudo-random points in the cube of half-width 0.8 a. */
export function selftestPoints(n: number, a: number): Float64Array {
  const out = new Float64Array(3 * n);
  let s = 12345;
  const rnd = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let i = 0; i < 3 * n; i++) out[i] = 0.8 * a * (2 * rnd() - 1);
  return out;
}

export interface Slice {
  name: 'xy' | 'xz' | 'yz';
  axes: [number, number];
  fixed: number;
  /** the value of the fixed coordinate */
  value: number;
  /** [lo, hi] along the two axes */
  extent: [[number, number], [number, number]];
  pts: Float64Array; // (3, ngrid^2), row-major over the grid with the first axis varying fastest
  /** 1 where the field is computed (inside the room, or outside the cymbal) */
  mask: Uint8Array;
}

/** Points on the three planes through x0; each grid runs from -a to a. */
export function slicePoints(ngrid: number, x0: ArrayLike<number>, a: number, p: number): { grid: Float64Array; slices: Slice[] } {
  const grid = new Float64Array(ngrid);
  for (let i = 0; i < ngrid; i++) grid[i] = -a + (2 * a * i) / (ngrid - 1);
  const defs: [Slice['name'], [number, number], number][] = [
    ['xy', [0, 1], 2],
    ['xz', [0, 2], 1],
    ['yz', [1, 2], 0],
  ];
  const slices: Slice[] = [];
  for (const [name, axes, fixed] of defs) {
    const N = ngrid * ngrid;
    const pts = new Float64Array(3 * N);
    const mask = new Uint8Array(N);
    for (let row = 0; row < ngrid; row++) {
      for (let col = 0; col < ngrid; col++) {
        const q = row * ngrid + col;
        const x = [0, 0, 0];
        x[axes[0]] = grid[col];
        x[axes[1]] = grid[row];
        x[fixed] = x0[fixed];
        pts[q] = x[0];
        pts[N + q] = x[1];
        pts[2 * N + q] = x[2];
        mask[q] = inside(x, a, p, 1e-3) ? 1 : 0;
      }
    }
    slices.push({ name, axes, fixed, value: x0[fixed], extent: [[-a, a], [-a, a]], pts, mask });
  }
  return { grid, slices };
}

/** Power radiated by the source, k/(4 pi) + Im v(x0), and absorbed by the walls, -k Re(lambda) int |u|^2 dS. */
export function energyBalance(k: number, lam: number, vAtSource: [number, number], wts: Float64Array, uWall: CVec) {
  const pSrc = k / (4 * Math.PI) + vAtSource[1];
  let s = 0;
  for (let i = 0; i < wts.length; i++) s += wts[i] * (uWall[2 * i] ** 2 + uWall[2 * i + 1] ** 2);
  const pWall = -k * lam * s;
  return { pSrc, pWall, mismatch: Math.abs(pSrc - pWall) / Math.abs(pSrc) };
}

/** Relative 2-norm difference of two complex vectors. */
export function relError(u: CVec, uex: CVec): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < u.length; i++) {
    num += (u[i] - uex[i]) ** 2;
    den += uex[i] ** 2;
  }
  return Math.sqrt(num / den);
}

/** Select the columns of x (3, n) where mask is set. */
export function maskedPoints(x: Float64Array, mask: Uint8Array): { pts: Float64Array; index: Int32Array } {
  const n = mask.length;
  let cnt = 0;
  for (let i = 0; i < n; i++) cnt += mask[i];
  const pts = new Float64Array(3 * cnt);
  const index = new Int32Array(cnt);
  let q = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    pts[q] = x[i];
    pts[cnt + q] = x[n + i];
    pts[2 * cnt + q] = x[2 * n + i];
    index[q++] = i;
  }
  return { pts, index };
}
