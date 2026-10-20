/**
 * Unrestarted GMRES for (zid I + K) x = rhs with the weighted inner product
 * <u, v> = sum conj(u) v w, following fmm3dbie's zgmres_guru. The Krylov
 * vectors and the Hessenberg system are kept in double precision on the CPU;
 * only the operator K is applied on the GPU.
 */
import type { CVec } from './problem.ts';

export type Apply = (x: CVec) => Promise<CVec>;

export interface GmresResult {
  x: CVec;
  niter: number;
  rres: number;
  residuals: number[];
}

export async function gmres(
  applyK: Apply,
  rhs: CVec,
  wts: Float64Array,
  zid: [number, number],
  eps: number,
  numit: number,
  onIter?: (it: number, rres: number) => void,
): Promise<GmresResult> {
  const n = wts.length;
  const wnorm = (v: CVec) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += (v[2 * i] ** 2 + v[2 * i + 1] ** 2) * wts[i];
    return Math.sqrt(s);
  };
  const rb = wnorm(rhs);
  const V: CVec[] = [];
  const m1 = numit + 1;
  const Hr = new Float64Array(m1 * numit);
  const Hi = new Float64Array(m1 * numit);
  const csr = new Float64Array(numit);
  const csi = new Float64Array(numit);
  const snr = new Float64Array(numit);
  const sni = new Float64Array(numit);
  const sr = new Float64Array(m1);
  const si = new Float64Array(m1);
  sr[0] = rb;
  const v0 = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) v0[i] = rhs[i] / rb;
  V.push(v0);
  const residuals: number[] = [];
  let itDone = 0;
  for (let it = 0; it < numit; it++) {
    let w = await applyK(V[it]);
    w = new Float64Array(w); // do not alias the operator's output
    // classical Gram-Schmidt against all previous vectors, repeated once
    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k <= it; k++) {
        const vk = V[k];
        let hr = 0;
        let hi = 0;
        for (let i = 0; i < n; i++) {
          const ar = vk[2 * i];
          const ai = vk[2 * i + 1];
          const br = w[2 * i] * wts[i];
          const bi = w[2 * i + 1] * wts[i];
          hr += ar * br + ai * bi; // conj(a) b
          hi += ar * bi - ai * br;
        }
        Hr[k * numit + it] += hr;
        Hi[k * numit + it] += hi;
        for (let i = 0; i < n; i++) {
          const vr = vk[2 * i];
          const vi = vk[2 * i + 1];
          w[2 * i] -= hr * vr - hi * vi;
          w[2 * i + 1] -= hr * vi + hi * vr;
        }
      }
    }
    Hr[it * numit + it] += zid[0];
    Hi[it * numit + it] += zid[1];
    const hn = wnorm(w);
    Hr[(it + 1) * numit + it] = hn;
    Hi[(it + 1) * numit + it] = 0;
    const vn = new Float64Array(2 * n);
    if (hn > 0) for (let i = 0; i < 2 * n; i++) vn[i] = w[i] / hn;
    V.push(vn);
    // apply the previous Givens rotations to the new column
    for (let k = 0; k < it; k++) {
      const ar = Hr[k * numit + it];
      const ai = Hi[k * numit + it];
      const br = Hr[(k + 1) * numit + it];
      const bi = Hi[(k + 1) * numit + it];
      // t1 = cs a + sn b ; t2 = -conj(sn) a + cs b
      const t1r = csr[k] * ar - csi[k] * ai + snr[k] * br - sni[k] * bi;
      const t1i = csr[k] * ai + csi[k] * ar + snr[k] * bi + sni[k] * br;
      const t2r = -(snr[k] * ar + sni[k] * ai) + csr[k] * br - csi[k] * bi;
      const t2i = -(snr[k] * ai - sni[k] * ar) + csr[k] * bi + csi[k] * br;
      Hr[k * numit + it] = t1r;
      Hi[k * numit + it] = t1i;
      Hr[(k + 1) * numit + it] = t2r;
      Hi[(k + 1) * numit + it] = t2i;
    }
    const ar = Hr[it * numit + it];
    const ai = Hi[it * numit + it];
    const br = Hr[(it + 1) * numit + it];
    const bi = Hi[(it + 1) * numit + it];
    const aa = Math.hypot(ar, ai);
    const r = Math.hypot(aa, Math.hypot(br, bi));
    if (r === 0) {
      csr[it] = 1;
      csi[it] = 0;
      snr[it] = 0;
      sni[it] = 0;
    } else {
      csr[it] = aa / r;
      csi[it] = 0;
      // sn = (a/|a|) conj(b) / r
      let pr = 1;
      let pi = 0;
      if (aa > 0) {
        pr = ar / aa;
        pi = ai / aa;
      }
      snr[it] = (pr * br + pi * bi) / r;
      sni[it] = (pi * br - pr * bi) / r;
    }
    // H[it, it] = cs a + sn b ; H[it+1, it] = 0
    Hr[it * numit + it] = csr[it] * ar - csi[it] * ai + snr[it] * br - sni[it] * bi;
    Hi[it * numit + it] = csr[it] * ai + csi[it] * ar + snr[it] * bi + sni[it] * br;
    Hr[(it + 1) * numit + it] = 0;
    Hi[(it + 1) * numit + it] = 0;
    // s[it+1] = -conj(sn) s[it] ; s[it] = cs s[it]
    const s0r = sr[it];
    const s0i = si[it];
    sr[it + 1] = -(snr[it] * s0r + sni[it] * s0i);
    si[it + 1] = -(snr[it] * s0i - sni[it] * s0r);
    sr[it] = csr[it] * s0r - csi[it] * s0i;
    si[it] = csr[it] * s0i + csi[it] * s0r;
    itDone = it + 1;
    const rres = Math.hypot(sr[it + 1], si[it + 1]) / rb;
    residuals.push(rres);
    onIter?.(itDone, rres);
    if (rres < eps) break;
  }
  // back substitution on the upper triangular system
  const yr = new Float64Array(itDone);
  const yi = new Float64Array(itDone);
  for (let i = itDone - 1; i >= 0; i--) {
    let tr = sr[i];
    let ti = si[i];
    for (let j = i + 1; j < itDone; j++) {
      const hr = Hr[i * numit + j];
      const hi = Hi[i * numit + j];
      tr -= hr * yr[j] - hi * yi[j];
      ti -= hr * yi[j] + hi * yr[j];
    }
    const hr = Hr[i * numit + i];
    const hi = Hi[i * numit + i];
    const d = hr * hr + hi * hi;
    yr[i] = (tr * hr + ti * hi) / d;
    yi[i] = (ti * hr - tr * hi) / d;
  }
  const x = new Float64Array(2 * n);
  for (let j = 0; j < itDone; j++) {
    const vj = V[j];
    for (let i = 0; i < n; i++) {
      const vr = vj[2 * i];
      const vi = vj[2 * i + 1];
      x[2 * i] += yr[j] * vr - yi[j] * vi;
      x[2 * i + 1] += yr[j] * vi + yi[j] * vr;
    }
  }
  const Kx = await applyK(x);
  const res = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    res[2 * i] = zid[0] * x[2 * i] - zid[1] * x[2 * i + 1] + Kx[2 * i] - rhs[2 * i];
    res[2 * i + 1] = zid[0] * x[2 * i + 1] + zid[1] * x[2 * i] + Kx[2 * i + 1] - rhs[2 * i + 1];
  }
  return { x, niter: itDone, rres: wnorm(res) / rb, residuals };
}
