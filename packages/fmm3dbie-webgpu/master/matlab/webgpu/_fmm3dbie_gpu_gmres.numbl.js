// Unrestarted GMRES for (zid I + K) x = rhs with the weighted inner product
// <u, v> = sum conj(u) v w, following fmm3dbie's zgmres_guru (ported from the
// WebGPU proof of concept's gmres.ts). Vectors are interleaved complex
// Float64Arrays; applyK(x) returns K x. Runs in numbl's worker, so applyK may
// combine GPU calls with other numbl calls (e.g. an FMM).

function gmres(applyK, rhs, wts, zid, eps, numit) {
  var n = wts.length;
  function wnorm(v) {
    var s = 0;
    for (var i = 0; i < n; i++) s += (v[2 * i] * v[2 * i] + v[2 * i + 1] * v[2 * i + 1]) * wts[i];
    return Math.sqrt(s);
  }
  var rb = wnorm(rhs);
  var V = [];
  var m1 = numit + 1;
  var Hr = new Float64Array(m1 * numit), Hi = new Float64Array(m1 * numit);
  var csr = new Float64Array(numit), csi = new Float64Array(numit);
  var snr = new Float64Array(numit), sni = new Float64Array(numit);
  var sr = new Float64Array(m1), si = new Float64Array(m1);
  sr[0] = rb;
  var v0 = new Float64Array(2 * n);
  for (var i = 0; i < 2 * n; i++) v0[i] = rhs[i] / rb;
  V.push(v0);
  var residuals = [];
  var itDone = 0;
  for (var it = 0; it < numit; it++) {
    var w = new Float64Array(applyK(V[it]));
    // classical Gram-Schmidt against all previous vectors, repeated once
    for (var pass = 0; pass < 2; pass++) {
      for (var k = 0; k <= it; k++) {
        var vk = V[k];
        var hr = 0, hi = 0;
        for (i = 0; i < n; i++) {
          var ar = vk[2 * i], ai = vk[2 * i + 1];
          var br = w[2 * i] * wts[i], bi = w[2 * i + 1] * wts[i];
          hr += ar * br + ai * bi; // conj(a) b
          hi += ar * bi - ai * br;
        }
        Hr[k * numit + it] += hr;
        Hi[k * numit + it] += hi;
        for (i = 0; i < n; i++) {
          var vr = vk[2 * i], vi = vk[2 * i + 1];
          w[2 * i] -= hr * vr - hi * vi;
          w[2 * i + 1] -= hr * vi + hi * vr;
        }
      }
    }
    Hr[it * numit + it] += zid[0];
    Hi[it * numit + it] += zid[1];
    var hn = wnorm(w);
    Hr[(it + 1) * numit + it] = hn;
    Hi[(it + 1) * numit + it] = 0;
    var vn = new Float64Array(2 * n);
    if (hn > 0) for (i = 0; i < 2 * n; i++) vn[i] = w[i] / hn;
    V.push(vn);
    // apply the previous Givens rotations to the new column
    for (k = 0; k < it; k++) {
      var a_r = Hr[k * numit + it], a_i = Hi[k * numit + it];
      var b_r = Hr[(k + 1) * numit + it], b_i = Hi[(k + 1) * numit + it];
      var t1r = csr[k] * a_r - csi[k] * a_i + snr[k] * b_r - sni[k] * b_i;
      var t1i = csr[k] * a_i + csi[k] * a_r + snr[k] * b_i + sni[k] * b_r;
      var t2r = -(snr[k] * a_r + sni[k] * a_i) + csr[k] * b_r - csi[k] * b_i;
      var t2i = -(snr[k] * a_i - sni[k] * a_r) + csr[k] * b_i + csi[k] * b_r;
      Hr[k * numit + it] = t1r;
      Hi[k * numit + it] = t1i;
      Hr[(k + 1) * numit + it] = t2r;
      Hi[(k + 1) * numit + it] = t2i;
    }
    var xr = Hr[it * numit + it], xi = Hi[it * numit + it];
    var yr0 = Hr[(it + 1) * numit + it], yi0 = Hi[(it + 1) * numit + it];
    var aa = Math.hypot(xr, xi);
    var r = Math.hypot(aa, Math.hypot(yr0, yi0));
    if (r === 0) {
      csr[it] = 1; csi[it] = 0; snr[it] = 0; sni[it] = 0;
    } else {
      csr[it] = aa / r;
      csi[it] = 0;
      var pr = 1, pi = 0;
      if (aa > 0) { pr = xr / aa; pi = xi / aa; }
      snr[it] = (pr * yr0 + pi * yi0) / r;
      sni[it] = (pi * yr0 - pr * yi0) / r;
    }
    Hr[it * numit + it] = csr[it] * xr - csi[it] * xi + snr[it] * yr0 - sni[it] * yi0;
    Hi[it * numit + it] = csr[it] * xi + csi[it] * xr + snr[it] * yi0 + sni[it] * yr0;
    Hr[(it + 1) * numit + it] = 0;
    Hi[(it + 1) * numit + it] = 0;
    var s0r = sr[it], s0i = si[it];
    sr[it + 1] = -(snr[it] * s0r + sni[it] * s0i);
    si[it + 1] = -(snr[it] * s0i - sni[it] * s0r);
    sr[it] = csr[it] * s0r - csi[it] * s0i;
    si[it] = csr[it] * s0i + csi[it] * s0r;
    itDone = it + 1;
    var rres = Math.hypot(sr[it + 1], si[it + 1]) / rb;
    residuals.push(rres);
    if (rres < eps) break;
  }
  // back substitution on the upper triangular system
  var y_r = new Float64Array(itDone), y_i = new Float64Array(itDone);
  for (i = itDone - 1; i >= 0; i--) {
    var tr = sr[i], ti = si[i];
    for (var j = i + 1; j < itDone; j++) {
      var h_r = Hr[i * numit + j], h_i = Hi[i * numit + j];
      tr -= h_r * y_r[j] - h_i * y_i[j];
      ti -= h_r * y_i[j] + h_i * y_r[j];
    }
    var d_r = Hr[i * numit + i], d_i = Hi[i * numit + i];
    var dd = d_r * d_r + d_i * d_i;
    y_r[i] = (tr * d_r + ti * d_i) / dd;
    y_i[i] = (ti * d_r - tr * d_i) / dd;
  }
  var x = new Float64Array(2 * n);
  for (j = 0; j < itDone; j++) {
    var vj = V[j];
    for (i = 0; i < n; i++) {
      x[2 * i] += y_r[j] * vj[2 * i] - y_i[j] * vj[2 * i + 1];
      x[2 * i + 1] += y_r[j] * vj[2 * i + 1] + y_i[j] * vj[2 * i];
    }
  }
  var Kx = applyK(x);
  var res = new Float64Array(2 * n);
  for (i = 0; i < n; i++) {
    res[2 * i] = zid[0] * x[2 * i] - zid[1] * x[2 * i + 1] + Kx[2 * i] - rhs[2 * i];
    res[2 * i + 1] = zid[0] * x[2 * i + 1] + zid[1] * x[2 * i] + Kx[2 * i + 1] - rhs[2 * i + 1];
  }
  return { x: x, niter: itDone, rres: wnorm(res) / rb, residuals: residuals };
}

return { gmres: gmres };
