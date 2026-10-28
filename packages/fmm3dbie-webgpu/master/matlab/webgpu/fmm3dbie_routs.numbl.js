// Stand-in for the fmm3dbie MEX gateway, which this package does not ship.
// A few small routines are ported here (from src/quadratures/near_field_routs.f,
// src/quadratures/far_field_routs.f90 and src/kernels/vpp.f90), so that the
// upstream MATLAB around them works:
//   get_rfacs, findnearmem, findnear, get_iquad_rsc   (getnear)
//   get_far_order                                     (get_oversampling_parameters)
//   vpp_eval_vec                                      (radpp_eval_fortran)
// Quad patches only (iptype 11). Other upstream functions that call the MEX
// fail with a message naming what the GPU backend does support.

function nums(v) {
  if (typeof v === "number") return [v];
  if (typeof v === "boolean") return [v ? 1 : 0];
  if (v && v.kind === "tensor") return v.data;
  if (v && v.kind === "complex_number") return [v.re];
  throw new RuntimeError("fmm3dbie_routs: expected a numeric argument");
}

function num(v) {
  return nums(v)[0];
}

function complexScalar(v) {
  if (typeof v === "number") return [v, 0];
  if (v && v.kind === "complex_number") return [v.re, v.im];
  if (v && v.kind === "tensor") return [v.data.length ? v.data[0] : 0, v.imag && v.imag.length ? v.imag[0] : 0];
  throw new RuntimeError("fmm3dbie_routs: expected a numeric wavenumber");
}

// an output with the shape of the corresponding input argument
function like(v, data) {
  var shape = v && v.kind === "tensor" ? v.shape.slice() : [1, 1];
  return RTV.tensor(Float64Array.from(data), shape);
}

// ── near_field_routs.f ─────────────────────────────────────────────────────

function getRfacs(norder) {
  var rfac = 1.25;
  if (norder <= 2) rfac = 2.75;
  if (norder <= 6 && norder > 2) rfac = 2.0;
  return [rfac, 1.25];
}

// targets i near patch j: |x_i - c_j|^2 <= rads_j^2, patches ascending
function nearLists(cms, rads, ndtarg, targs, ntarg) {
  var npatches = rads.length;
  var rowPtr = [1], colInd = [];
  for (var i = 0; i < ntarg; i++) {
    var x = targs[ndtarg * i], y = targs[ndtarg * i + 1], z = targs[ndtarg * i + 2];
    for (var j = 0; j < npatches; j++) {
      var dx = cms[3 * j] - x, dy = cms[3 * j + 1] - y, dz = cms[3 * j + 2] - z;
      if (dx * dx + dy * dy + dz * dz <= rads[j] * rads[j]) colInd.push(j + 1);
    }
    rowPtr.push(colInd.length + 1);
  }
  return { rowPtr: rowPtr, colInd: colInd };
}

// ── far_field_routs.f90: get_far_order ────────────────────────────────────

var NFARS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16];
var OVER4PI = 0.07957747154594767;

// Gauss-Legendre nodes (ascending) and weights on [-1, 1], n points
function leggauss(n) {
  var x = new Float64Array(n), w = new Float64Array(n);
  for (var i = 0; i < n; i++) {
    var t = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5)), dp = 1;
    for (var it = 0; it < 100; it++) {
      var p0 = 1, p1 = t;
      for (var k = 2; k <= n; k++) { var p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = (n * (t * p1 - p0)) / (t * t - 1);
      var dt = p1 / dp;
      t -= dt;
      if (Math.abs(dt) < 1e-15) break;
    }
    x[n - 1 - i] = t;
    w[n - 1 - i] = 2 / ((1 - t * t) * dp * dp);
  }
  return { x: x, w: w };
}

function legendre(x, n) {
  var P = new Float64Array(n + 1);
  P[0] = 1;
  if (n >= 1) P[1] = x;
  for (var k = 1; k < n; k++) P[k + 1] = ((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1);
  return P;
}

// tensor nodes (u fastest) and weights of order n, and the order-norder
// basis P_a(u) P_b(v) (index b*(norder+1) + a) at them: pmat[l*nn + j]
function discNodes(n, norder) {
  var g = leggauss(n + 1);
  var n1 = n + 1, nn = n1 * n1, m1 = norder + 1, npols = m1 * m1;
  var w = new Float64Array(nn), pmat = new Float64Array(npols * nn);
  for (var b = 0; b < n1; b++) {
    for (var a = 0; a < n1; a++) {
      var j = b * n1 + a;
      w[j] = g.w[a] * g.w[b];
      var Pu = legendre(g.x[a], norder), Pv = legendre(g.x[b], norder);
      for (var q = 0; q < m1; q++) for (var p = 0; p < m1; p++) pmat[(q * m1 + p) * nn + j] = Pu[p] * Pv[q];
    }
  }
  return { nn: nn, w: w, pmat: pmat };
}

// srctmp (12 per node: xyz, r_u, r_v, unit normal) and quadrature weights
function nodesGeometry(coefs, npols, d) {
  var src = new Float64Array(12 * d.nn), qw = new Float64Array(d.nn);
  for (var j = 0; j < d.nn; j++) {
    for (var c = 0; c < 9; c++) {
      var s = 0;
      for (var l = 0; l < npols; l++) s += coefs[9 * l + c] * d.pmat[l * d.nn + j];
      src[12 * j + c] = s;
    }
    var u = src.subarray(12 * j + 3, 12 * j + 6), v = src.subarray(12 * j + 6, 12 * j + 9);
    var cx = u[1] * v[2] - u[2] * v[1], cy = u[2] * v[0] - u[0] * v[2], cz = u[0] * v[1] - u[1] * v[0];
    var rr = Math.sqrt(cx * cx + cy * cy + cz * cz);
    src[12 * j + 9] = cx / rr; src[12 * j + 10] = cy / rr; src[12 * j + 11] = cz / rr;
    qw[j] = d.w[j] * rr;
  }
  return { src: src, qw: qw };
}

// h3d_slp (-1), h3d_dlp (0), h3d_qlp (1) of helm_kernels.f90, complex result
function helmKernel(order, s, j, t, zr, zi) {
  var dx = t[0] - s[12 * j], dy = t[1] - s[12 * j + 1], dz = t[2] - s[12 * j + 2];
  var rsq = dx * dx + dy * dy + dz * dz, r = Math.sqrt(rsq);
  // e = exp(i zk r)
  var ea = Math.exp(-zi * r), er = ea * Math.cos(zr * r), ei = ea * Math.sin(zr * r);
  if (order === -1) return [(er / r) * OVER4PI, (ei / r) * OVER4PI];
  var d = dx * s[12 * j + 9] + dy * s[12 * j + 10] + dz * s[12 * j + 11];
  var ar, ai; // the factor multiplying exp(i zk r) / r^3
  if (order === 0) {
    // d (1 - i zk r)
    ar = d * (1 + zi * r); ai = -d * zr * r;
  } else if (order === 1) {
    // d^2 (-(i zk)^2 - 3/r^2 + 3 i zk / r) + 1 - i zk r
    var ik2r = zr * zr - zi * zi, ik2i = 2 * zr * zi; // -(i zk)^2 = zk^2
    ar = d * d * (ik2r - 3 / rsq - (3 * zi) / r) + 1 + zi * r;
    ai = d * d * (ik2i + (3 * zr) / r) - zr * r;
  } else {
    throw new RuntimeError("fmm3dbie_routs: get_far_order supports kernel orders -1, 0, 1");
  }
  var f = OVER4PI / (rsq * r);
  return [(ar * er - ai * ei) * f, (ar * ei + ai * er) * f];
}

// LAPACK dlaran
function dlaran(iseed) {
  var M1 = 494, M2 = 322, M3 = 2508, M4 = 2549, IPW2 = 4096, R = 1 / IPW2, out;
  do {
    var it4 = iseed[3] * M4;
    var it3 = Math.floor(it4 / IPW2);
    it4 -= IPW2 * it3;
    it3 += iseed[2] * M4 + iseed[3] * M3;
    var it2 = Math.floor(it3 / IPW2);
    it3 -= IPW2 * it2;
    it2 += iseed[1] * M4 + iseed[2] * M3 + iseed[3] * M2;
    var it1 = Math.floor(it2 / IPW2);
    it2 -= IPW2 * it1;
    it1 += iseed[0] * M4 + iseed[1] * M3 + iseed[2] * M2 + iseed[3] * M1;
    it1 %= IPW2;
    iseed[0] = it1; iseed[1] = it2; iseed[2] = it3; iseed[3] = it4;
    out = R * (it1 + R * (it2 + R * (it3 + R * it4)));
  } while (out === 1);
  return out;
}

// the smallest order in NFARS at which the patch's near targets (and points
// on a sphere of radius rad around cm) see the basis integrals converged
function farOrderGuru(eps, norder, coefs, cm, rad, ikerorder, zr, zi, ndtarg, targs, tlist) {
  var ntarg = tlist.length, nmax = 25;
  var m1 = norder + 1, npols = m1 * m1;
  var dd = tlist.map(function (it) {
    var s = 0;
    for (var c = 0; c < 3; c++) s += Math.pow(targs[ndtarg * it + c] - cm[c], 2);
    return { d: s, it: it };
  });
  dd.sort(function (p, q) { return p.d - q.d; });
  var tt = [];
  var i0 = Math.min(10, Math.floor(ntarg / 2));
  for (var i = 1; i <= i0; i++) {
    var ii = dd[ntarg - i].it;
    tt.push([targs[ndtarg * ii], targs[ndtarg * ii + 1], targs[ndtarg * ii + 2]]);
  }
  for (i = i0 + 1; i <= nmax; i++) {
    var phi = dlaran([0, 0, 0, (2 * (1001 + i) + 1) % 4096]) * 2 * Math.PI;
    var thet = dlaran([0, 0, 0, (2 * (3042 + i) + 1) % 4096]) * Math.PI;
    tt.push([cm[0] + rad * Math.sin(thet) * Math.cos(phi), cm[1] + rad * Math.sin(thet) * Math.sin(phi), cm[2] + rad * Math.cos(thet)]);
  }
  // epsp from the smallest area element at the patch's own nodes
  var dOwn = discNodes(norder, norder), own = nodesGeometry(coefs, npols, dOwn);
  var rsc = Infinity;
  for (var j = 0; j < dOwn.nn; j++) {
    var jac = own.qw[j] / dOwn.w[j];
    rsc = Math.min(rsc, jac * jac);
  }
  var epsp = Math.pow(rsc, 0.25) * eps;
  function integrals(nfar) {
    var d = discNodes(nfar, norder), g = nodesGeometry(coefs, npols, d);
    var v = new Float64Array(2 * npols * nmax);
    for (var t = 0; t < nmax; t++) {
      for (var j = 0; j < d.nn; j++) {
        var k = helmKernel(ikerorder, g.src, j, tt[t], zr, zi);
        for (var l = 0; l < npols; l++) {
          var f = d.pmat[l * d.nn + j] * g.qw[j];
          v[2 * (t * npols + l)] += k[0] * f;
          v[2 * (t * npols + l) + 1] += k[1] * f;
        }
      }
    }
    return v;
  }
  var vcomp = integrals(NFARS[0]);
  var nnn;
  for (nnn = 1; nnn < NFARS.length; nnn++) {
    var vref = integrals(NFARS[nnn]);
    var errmax = 0;
    for (var t = 0; t < nmax; t++) {
      var e2 = 0;
      for (var l = 0; l < npols; l++) {
        var a = 2 * (t * npols + l);
        e2 += Math.pow(vref[a] - vcomp[a], 2) + Math.pow(vref[a + 1] - vcomp[a + 1], 2);
      }
      errmax = Math.max(errmax, Math.sqrt(e2));
    }
    if (errmax < epsp) break;
    vcomp = vref;
  }
  return NFARS[Math.max(nnn - 1, 0)];
}

function getFarOrder(a) {
  var eps = num(a[1]), npatches = num(a[2]);
  var norders = nums(a[3]), ixyzs = nums(a[4]), iptype = nums(a[5]);
  var cms = nums(a[6]), rads = nums(a[7]), srccoefs = nums(a[9]);
  var ndtarg = num(a[10]), ntarg = num(a[11]), targs = nums(a[12]);
  var ikerorder = num(a[13]), zk = complexScalar(a[14]);
  var rowPtr = nums(a[16]), colInd = nums(a[17]), rfac = num(a[18]);
  // targets near each patch (the csc form of row_ptr / col_ind)
  var near = [];
  for (var j = 0; j < npatches; j++) near.push([]);
  for (var i = 0; i < ntarg; i++) for (var p = rowPtr[i] - 1; p < rowPtr[i + 1] - 1; p++) near[colInd[p] - 1].push(i);
  var novers = new Float64Array(npatches), ixyzso = new Float64Array(npatches + 1);
  ixyzso[0] = 1;
  for (j = 0; j < npatches; j++) {
    if (iptype[j] !== 11) throw new RuntimeError("fmm3dbie_routs: get_far_order supports quad patches (iptype 11) only");
    var i1 = ixyzs[j] - 1, i2 = ixyzs[j + 1] - 1;
    var coefs = srccoefs.subarray ? srccoefs.subarray(9 * i1, 9 * i2) : srccoefs.slice(9 * i1, 9 * i2);
    novers[j] = farOrderGuru(eps, norders[j], coefs, [cms[3 * j], cms[3 * j + 1], cms[3 * j + 2]], rfac * rads[j], ikerorder, zk[0], zk[1], ndtarg, targs, near[j]);
    ixyzso[j + 1] = ixyzso[j] + (novers[j] + 1) * (novers[j] + 1);
  }
  return [like(a[19], novers), like(a[20], ixyzso)];
}

// ── vpp.f90: vpp_eval_vec ──────────────────────────────────────────────────

function vppEvalVec(a) {
  var nr = num(a[1]), r = nums(a[2]), dpars = nums(a[4]), ipars = nums(a[6]);
  var nbin = ipars[0], n = ipars[1], ibins = ipars[2] - 1, icfs = ipars[3] - 1, nv = ipars[4];
  var vals = new Float64Array(nv * nr);
  for (var i = 0; i < nr; i++) {
    // ipp_getbin
    var x = r[i], ibin = 1, jbin = nbin, lo = dpars[ibins], hi = dpars[ibins + nbin - 1];
    if (x >= lo) {
      if (x >= hi) { ibin = nbin; lo = hi; }
      else {
        while (jbin > ibin + 1) {
          var mid = Math.floor((ibin + jbin) / 2), am = dpars[ibins + mid - 1];
          if (x >= am) { lo = am; ibin = mid; } else jbin = mid;
        }
      }
    }
    // vpp_eval0: coefficients (nv, n), highest degree first
    var c0 = icfs + n * nv * (ibin - 1), h = x - lo;
    for (var k = 0; k < nv; k++) {
      var v = dpars[c0 + k];
      for (var m = 1; m < n; m++) v = dpars[c0 + m * nv + k] + h * v;
      vals[nv * i + k] = v;
    }
  }
  return [like(a[7], vals)];
}

function dispatch(routine, a) {
  switch (routine) {
    case "get_rfacs":
      var rf = getRfacs(num(a[1]));
      return [RTV.num(rf[0]), RTV.num(rf[1])];
    case "findnearmem": {
      var L = nearLists(nums(a[1]), nums(a[3]), num(a[4]), nums(a[5]), num(a[6]));
      return [RTV.num(L.colInd.length)];
    }
    case "findnear": {
      var L2 = nearLists(nums(a[1]), nums(a[3]), num(a[4]), nums(a[5]), num(a[6]));
      return [like(a[7], L2.rowPtr), like(a[8], L2.colInd)];
    }
    case "get_iquad_rsc": {
      var ixyzs = nums(a[2]), colInd = nums(a[6]), nnz = num(a[4]);
      var iquad = new Float64Array(nnz + 1);
      iquad[0] = 1;
      for (var i = 0; i < nnz; i++) iquad[i + 1] = iquad[i] + ixyzs[colInd[i]] - ixyzs[colInd[i] - 1];
      return [like(a[7], iquad)];
    }
    case "get_far_order":
      return getFarOrder(a);
    case "vpp_eval_vec":
      return vppEvalVec(a);
  }
  throw new RuntimeError(
    "fmm3dbie-webgpu: " + routine + " needs the fmm3dbie MEX, which this package does not have. " +
      "On the GPU, helm3d.impedance.{solver,eval} and helm3d.neumann.{solver,eval} are supported; " +
      "use quad_sphere for geometry."
  );
}

register(function (args, nargout) {
  var id = args.length > 0 && args[0] && args[0].kind === "char" ? args[0].value : "";
  var out = dispatch(id.split("(")[0] || "this routine", args);
  return nargout <= 1 ? out[0] : out.slice(0, nargout);
});
