// The operator K of fmm3dbie's helm_rpcomb_imp equation, applied on the numbl
// side from GPU passes (fmm3dbie_gpu backend, surface_applyAt), with the far
// field either from the GPU's dense sum (far = 'dense') or from an FMM
// (far = 'fmm': hfmm3d from the fmm3d package, called through a function
// handle). Ported from the WebGPU proof of concept's Operator.applyK:
//
//   K s = S'_k s + i a S'_kap phi2 + i a (D'_k - D'_kap) phi1
//         + i k lam [S_k s + i a D_k phi1 + i a w phi1],
//   phi1 = S_kap s, phi2 = S'_kap s, kap = i|k|, w = (-1)^ifinout / 2,
//
// with kernels numbered 0: S'_k, 1: S_kap, 2: S'_kap, 3: D'_k - D'_kap,
// 4: S_k, 5: D_k. All vectors are interleaved complex Float64Arrays.

// info: from surface_info; geo: from surface_farGeometry (only for 'fmm')
function makeOperator(id, info, geo, far, hfmm, epsFmm) {
  var npts = info.wts.length;
  var k = info.k, a = info.alpha, sgn = info.ifinout % 2 === 0 ? 1 : -1;
  var fmm = far === "fmm" ? makeFmm(info, geo, hfmm, epsFmm) : null;
  var gpuFar = far === "fmm" ? "none" : "dense";

  function pass(qa, qb, sel, reqs) {
    return gpu.call("fmm3dbie_gpu", "surface_applyAt", { id: id, qa: qa, qb: qb, sel: sel, reqs: reqs, far: gpuFar }).F;
  }
  function get(F, m, i) {
    return [F[2 * (m * npts + i)], F[2 * (m * npts + i) + 1]];
  }

  return function applyK(lam, x) {
    var F = pass(x, null, [1, 1, 1, 0, 1, 0], [[0, 0], [1, 0], [2, 0], [4, 0]]);
    var f1 = fmm ? fmm.pass1(x) : null; // S'_k, S_kap, S'_kap, S_k of x
    var phi1 = new Float64Array(2 * npts), phi2 = new Float64Array(2 * npts), res = new Float64Array(2 * npts);
    for (var i = 0; i < npts; i++) {
      var s0 = get(F, 0, i), s1 = get(F, 1, i), s2 = get(F, 2, i), s4 = get(F, 4, i);
      if (f1) {
        s0[0] += f1.spk[2 * i]; s0[1] += f1.spk[2 * i + 1];
        s1[0] += f1.skap[2 * i]; s1[1] += f1.skap[2 * i + 1];
        s2[0] += f1.spkap[2 * i]; s2[1] += f1.spkap[2 * i + 1];
        s4[0] += f1.sk[2 * i]; s4[1] += f1.sk[2 * i + 1];
      }
      phi1[2 * i] = s1[0]; phi1[2 * i + 1] = s1[1];
      phi2[2 * i] = s2[0]; phi2[2 * i + 1] = s2[1];
      // res = F0 + i k lam F4
      res[2 * i] = s0[0] - k * lam * s4[1];
      res[2 * i + 1] = s0[1] + k * lam * s4[0];
    }
    var G = pass(phi1, phi2, [0, 0, 2, 1, 0, 1], [[2, 1], [3, 0], [5, 0]]);
    var f2 = fmm ? fmm.pass2(phi1, phi2) : null; // S'_kap phi2 + (D'_k - D'_kap) phi1, D_k phi1
    for (i = 0; i < npts; i++) {
      var g2 = get(G, 2, i), g3 = get(G, 3, i), g5 = get(G, 5, i);
      var g23r = g2[0] + g3[0], g23i = g2[1] + g3[1];
      if (f2) {
        g23r += f2.g23[2 * i]; g23i += f2.g23[2 * i + 1];
        g5[0] += f2.dk[2 * i]; g5[1] += f2.dk[2 * i + 1];
      }
      // res += i a (G2 + G3) - k a lam G5 + sgn k a lam phi1 / 2
      res[2 * i] += -a * g23i - k * a * lam * g5[0] + (sgn * k * a * lam * phi1[2 * i]) / 2;
      res[2 * i + 1] += a * g23r - k * a * lam * g5[1] + (sgn * k * a * lam * phi1[2 * i + 1]) / 2;
    }
    return res;
  };
}

var FOUR_PI = 4 * Math.PI;

// The far field over all oversampled nodes by hfmm3d. Its Helmholtz kernel
// is exp(ikr)/(4 pi r) (as fmm3dbie's helm3d.fmm uses it), with dipoles
// giving dG/dn_y, so charges q w and dipoles n q w give S and D. From the
// FMM's result we subtract, in double precision, the smooth rule on each
// target's near patches, which the GPU's near quadrature replaces (the
// "add and subtract" of fmm3dbie). Some oversampled nodes lie very close to
// a target on its own patch, so this cannot be done in the GPU's f32.
function makeFmm(info, geo, hfmm, epsFmm) {
  var npts = info.wts.length;
  var N = geo.w.length, npols = geo.npols, npolsO = geo.npolsO;
  var sv = info.srcvals; // (12, npts) row-major
  var src = new Float64Array(3 * N), targ = new Float64Array(3 * npts), tn = new Float64Array(3 * npts);
  for (var j = 0; j < N; j++) for (var d = 0; d < 3; d++) src[3 * j + d] = geo.x[d * N + j];
  for (var i = 0; i < npts; i++) {
    for (d = 0; d < 3; d++) {
      targ[3 * i + d] = sv[d * npts + i];
      tn[3 * i + d] = sv[(9 + d) * npts + i];
    }
  }
  var srcT = RTV.tensor(src, [3, N]);
  var targT = RTV.tensor(targ, [3, npts]);
  var rowPtr = geo.rowPtr, pairP = geo.pairP;
  var k = info.k, kap = Math.abs(k);

  // values at the oversampled nodes, times the smooth weights
  function over(q) {
    var re = new Float64Array(N), im = new Float64Array(N);
    for (var p = 0; p < geo.npatches; p++) {
      for (var kk = 0; kk < npolsO; kk++) {
        var sr = 0, si = 0;
        for (var m = 0; m < npols; m++) {
          var M = geo.overMat[kk * npols + m];
          sr += M * q[2 * (p * npols + m)];
          si += M * q[2 * (p * npols + m) + 1];
        }
        var s = p * npolsO + kk, wf = geo.w[s];
        re[s] = sr * wf;
        im[s] = si * wf;
      }
    }
    return { re: re, im: im };
  }
  function dipoles(c, sign) {
    var re = new Float64Array(3 * N), im = new Float64Array(3 * N);
    for (var s = 0; s < N; s++) {
      for (var d = 0; d < 3; d++) {
        var nd = sign * geo.n[d * N + s];
        re[3 * s + d] = nd * c.re[s];
        im[3 * s + d] = nd * c.im[s];
      }
    }
    return RTV.tensor(re, [3, N], im);
  }
  // The smooth rule on the near patches: potential and normal derivative at
  // the surface nodes of charges c and dipoles sign n c2 (either may be null),
  // with wavenumber kr + i ki.
  function nearSmooth(kr, ki, c, c2, sign, pot, dn) {
    var X = geo.x, Nn = geo.n;
    for (var i = 0; i < npts; i++) {
      var x0 = targ[3 * i], x1 = targ[3 * i + 1], x2 = targ[3 * i + 2];
      var n0 = tn[3 * i], n1 = tn[3 * i + 1], n2 = tn[3 * i + 2];
      var pr = 0, pi = 0, dr = 0, di = 0;
      for (var p = rowPtr[i]; p < rowPtr[i + 1]; p++) {
        var s0 = pairP[p] * npolsO;
        for (var s = s0; s < s0 + npolsO; s++) {
          var d0 = x0 - X[s], d1 = x1 - X[N + s], d2 = x2 - X[2 * N + s];
          var r2 = d0 * d0 + d1 * d1 + d2 * d2, r = Math.sqrt(r2);
          // G = exp(ikr) / (4 pi r), z = ikr
          var zr = -ki * r, zi = kr * r;
          var ea = Math.exp(zr) / (FOUR_PI * r);
          var gr = ea * Math.cos(zi), gi = ea * Math.sin(zi);
          var nd = (n0 * d0 + n1 * d1 + n2 * d2) / r2;
          // A = (z - 1) G
          var ar = (zr - 1) * gr - zi * gi, ai = (zr - 1) * gi + zi * gr;
          if (c) {
            var cr = c.re[s], ci = c.im[s];
            pr += cr * gr - ci * gi; pi += cr * gi + ci * gr;
            dr += nd * (cr * ar - ci * ai); di += nd * (cr * ai + ci * ar);
          }
          if (c2) {
            var er = sign * c2.re[s], ei = sign * c2.im[s];
            var ns0 = Nn[s], ns1 = Nn[N + s], ns2 = Nn[2 * N + s];
            var vd = (ns0 * d0 + ns1 * d1 + ns2 * d2) / r2;
            var nv = (n0 * ns0 + n1 * ns1 + n2 * ns2) / r2;
            // dG/dn_y = -(z - 1) G (n_s . d) / r^2
            pr -= vd * (er * ar - ei * ai); pi -= vd * (er * ai + ei * ar);
            // B = (3 - 3z + z^2) G; d/dn_t dG/dn_y = -(A nv + B nd vd)
            var br3 = 3 - 3 * zr + zr * zr - zi * zi, bi3 = -3 * zi + 2 * zr * zi;
            var br = br3 * gr - bi3 * gi, bi = br3 * gi + bi3 * gr;
            var hr = ar * nv + br * nd * vd, hi = ai * nv + bi * nd * vd;
            dr -= er * hr - ei * hi; di -= er * hi + ei * hr;
          }
        }
      }
      pot[2 * i] -= pr; pot[2 * i + 1] -= pi;
      dn[2 * i] -= dr; dn[2 * i + 1] -= di;
    }
  }
  // the far field: potential and normal derivative at the surface nodes,
  // interleaved complex, of charges c and dipoles sign n c2
  function run(imagK, c, c2, sign) {
    var fields = { sources: srcT };
    if (c) fields.charges = RTV.tensor(c.re, [1, N], c.im);
    if (c2) fields.dipoles = dipoles(c2, sign);
    var zk = imagK ? RTV.complex(0, kap) : k;
    var U = callHandle(hfmm, [epsFmm, zk, RTV.struct(fields), 0, targT, 2]);
    var pot = U.fields.get("pottarg"), grad = U.fields.get("gradtarg");
    var p = new Float64Array(2 * npts), dn = new Float64Array(2 * npts);
    var gim = grad.imag, pim = pot.imag;
    for (var i = 0; i < npts; i++) {
      p[2 * i] = pot.data[i];
      p[2 * i + 1] = pim ? pim[i] : 0;
      var r = 0, im = 0;
      for (var d = 0; d < 3; d++) {
        r += tn[3 * i + d] * grad.data[3 * i + d];
        im += tn[3 * i + d] * (gim ? gim[3 * i + d] : 0);
      }
      dn[2 * i] = r;
      dn[2 * i + 1] = im;
    }
    nearSmooth(imagK ? 0 : k, imagK ? kap : 0, c, c2, sign, p, dn);
    return { pot: p, dn: dn };
  }
  return {
    pass1: function (x) {
      var c = over(x);
      var a = run(false, c, null, 0);
      var b = run(true, c, null, 0);
      return { spk: a.dn, sk: a.pot, skap: b.pot, spkap: b.dn };
    },
    pass2: function (phi1, phi2) {
      var c1 = over(phi1), c2 = over(phi2);
      var a = run(false, null, c1, 1); // D_k phi1, D'_k phi1
      var b = run(true, c2, c1, -1); // S'_kap phi2 - D'_kap phi1
      var g23 = new Float64Array(2 * npts);
      for (var i = 0; i < 2 * npts; i++) g23[i] = b.dn[i] + a.dn[i];
      return { g23: g23, dk: a.pot };
    },
  };
}

// For testing: both passes on given densities, far field dense vs FMM;
// returns the relative difference per kernel (pass 1: 0, 1, 2, 4 of x;
// pass 2: 2 + 3 and 5 of phi1, phi2).
function comparePasses(id, info, geo, hfmm, epsFmm, x, phi1, phi2) {
  var npts = info.wts.length;
  var fmm = makeFmm(info, geo, hfmm, epsFmm);
  function pass(qa, qb, sel, reqs, far) {
    return gpu.call("fmm3dbie_gpu", "surface_applyAt", { id: id, qa: qa, qb: qb, sel: sel, reqs: reqs, far: far }).F;
  }
  function rel(a, b) {
    var num = 0, den = 0;
    for (var i = 0; i < a.length; i++) { num = Math.max(num, Math.abs(a[i] - b[i])); den = Math.max(den, Math.abs(b[i])); }
    return num / den;
  }
  function slot(F, m) { return Array.prototype.slice.call(F, 2 * m * npts, 2 * (m + 1) * npts); }
  var Fd = pass(x, null, [1, 1, 1, 0, 1, 0], [[0, 0], [1, 0], [2, 0], [4, 0]], "dense");
  var Fs = pass(x, null, [1, 1, 1, 0, 1, 0], [[0, 0], [1, 0], [2, 0], [4, 0]], "none");
  var f1 = fmm.pass1(x);
  var out = [];
  var add = function (F, m, v) { var a = slot(F, m); for (var i = 0; i < a.length; i++) a[i] += v[i]; return a; };
  out.push(rel(add(Fs, 0, f1.spk), slot(Fd, 0)));
  out.push(rel(add(Fs, 1, f1.skap), slot(Fd, 1)));
  out.push(rel(add(Fs, 2, f1.spkap), slot(Fd, 2)));
  out.push(rel(add(Fs, 4, f1.sk), slot(Fd, 4)));
  var Gd = pass(phi1, phi2, [0, 0, 2, 1, 0, 1], [[2, 1], [3, 0], [5, 0]], "dense");
  var Gs = pass(phi1, phi2, [0, 0, 2, 1, 0, 1], [[2, 1], [3, 0], [5, 0]], "none");
  var f2 = fmm.pass2(phi1, phi2);
  var g23d = slot(Gd, 2), g3d = slot(Gd, 3), g23s = slot(Gs, 2), g3s = slot(Gs, 3);
  for (var i = 0; i < g23d.length; i++) { g23d[i] += g3d[i]; g23s[i] += g3s[i] + f2.g23[i]; }
  out.push(rel(g23s, g23d));
  out.push(rel(add(Gs, 5, f2.dk), slot(Gd, 5)));
  return out;
}

return { makeOperator: makeOperator, comparePasses: comparePasses };
