// GPU backend for the fmm3dbie subset (needs numbl's `gpu` global; see
// numbl src/gpuHelper.ts). The GPU side is backend/*.ts, bundled by
// backend/build.mjs into _fmm3dbie_gpu_backend.numbl.js. Single precision on
// the GPU; zk real. Objects live behind numeric handles until freed, across
// runs.
//
// Point sums (far field only):
//   h = fmm3dbie_gpu('create', zk, src, srcn, targ)
//     sources src (3,ns) with normals srcn (3,ns), targets targ (3,nt)
//   pot = fmm3dbie_gpu('apply', h, kind, q)
//     pot(i) = sum_j K(targ(:,i), src(:,j)) q(j), pot (1,nt), with
//     kind 'slp': K = G_k(x,y) = exp(1i*zk*r)/(4*pi*r)
//     kind 'dlp': K = dG_k/dn_y = n_y.(x-y) (1 - 1i*zk*r) exp(1i*zk*r)/(4*pi*r^3)
//     Pairs with r = 0 are skipped.
//   pot = fmm3dbie_gpu('h3d_slp', zk, src, q, targ)   one shot
//
// Layer potentials on a surface of quad patches (iptype 11, one order), with
// near quadrature:
//   [h, info] = fmm3dbie_gpu('surface_create', norder, srcvals, zk, alpha, lam, ifinout)
//     srcvals (12,npts) as from extract_arrays(S); alpha, lam, ifinout as in
//     fmm3dbie's helm_rpcomb_imp (they do not affect surface_apply)
//   v = fmm3dbie_gpu('surface_apply', h, kernel, sigma)
//     principal value on the surface of the layer potential with density sigma
//     (npts), v (1,npts); kernel 's', 'd', 'sprime', 's_kappa',
//     'sprime_kappa' or 'dprime_diff' (D'_k - D'_kappa), kappa = 1i*abs(zk)
//
//   [sigma, phi1, errs, rres] = fmm3dbie_gpu('surface_solve', h, rhs, eps_gmres, maxit, lam[, far, hfmm])
//     GMRES for fmm3dbie's helm_rpcomb_imp equation du/dn + 1i*zk*lam*u = rhs,
//     u = S_k[sigma] + 1i*alpha*D_k[phi1], phi1 = S_kappa[sigma]; lam real
//     scalar. sigma, phi1 (1,npts); errs (niter,1) relative residuals.
//     far = 'dense' (default): far field by the dense GPU sum; far = 'fmm':
//     by hfmm3d (pass @hfmm3d as hfmm), minus the smooth rule on near
//     patches (in double precision, here).
//   u = fmm3dbie_gpu('surface_eval', h, targ, sigma, phi1)
//     u = S_k[sigma] + 1i*alpha*D_k[phi1] at off-surface targets targ (3,n);
//     u (n,1)
//
//   fmm3dbie_gpu('free', h), fmm3dbie_gpu('freeall'), n = fmm3dbie_gpu('count')

if (gpu) gpu.load("fmm3dbie_gpu", importJS("_fmm3dbie_gpu_backend"));

// ── numbl side ──────────────────────────────────────────────────────────────

function realScalar(v, what) {
  if (typeof v === "number") return v;
  if (v && v.kind === "tensor" && v.data.length === 1 && !v.imag) return v.data[0];
  throw new RuntimeError("fmm3dbie_gpu: " + what + " must be a real scalar");
}

function str(v, what) {
  if (v && v.kind === "char") return v.value;
  if (v && v.kind === "string") return v.value;
  throw new RuntimeError("fmm3dbie_gpu: " + what + " must be a string");
}

function points(v, what) {
  if (!v || v.kind !== "tensor" || v.shape[0] !== 3 || v.imag)
    throw new RuntimeError("fmm3dbie_gpu: " + what + " must be a real 3-by-n array");
  return v.data;
}

// Interleaved (re, im) of a numeric vector, as a Float64Array.
function complexInterleaved(v, what) {
  var re, im;
  if (typeof v === "number") { re = [v]; im = null; }
  else if (v && v.kind === "complex_number") { re = [v.re]; im = [v.im]; }
  else if (v && v.kind === "tensor") { re = v.data; im = v.imag; }
  else throw new RuntimeError("fmm3dbie_gpu: " + what + " must be numeric");
  var out = new Float64Array(2 * re.length);
  for (var i = 0; i < re.length; i++) { out[2 * i] = re[i]; out[2 * i + 1] = im ? im[i] : 0; }
  return out;
}

// Sources (position relative to the source centroid c, computed in double so
// that single precision spends its digits on the geometry, then normal) as
// two vec4f each; targets as one vec4f each.
function packSources(xyz, nrm, c) {
  var n = xyz.length / 3;
  var out = new Float32Array(8 * n);
  for (var i = 0; i < n; i++) {
    for (var d = 0; d < 3; d++) {
      out[8 * i + d] = xyz[3 * i + d] - c[d];
      out[8 * i + 4 + d] = nrm ? nrm[3 * i + d] : 0;
    }
  }
  return out;
}

function packTargets(xyz, c) {
  var n = xyz.length / 3;
  var out = new Float32Array(4 * n);
  for (var i = 0; i < n; i++) for (var d = 0; d < 3; d++) out[4 * i + d] = xyz[3 * i + d] - c[d];
  return out;
}

function centroid(xyz) {
  var n = xyz.length / 3;
  var c = [0, 0, 0];
  for (var j = 0; j < n; j++) for (var d = 0; d < 3; d++) c[d] += xyz[3 * j + d] / n;
  return c;
}

// No handle bookkeeping on this side: numbl may re-evaluate this file between
// runs, while the GPU-side state persists.
function create(zk, src, nrm, targ) {
  var c = centroid(src);
  return gpu.call("fmm3dbie_gpu", "points_create", { zk: zk, src: packSources(src, nrm, c), targ: packTargets(targ, c) }).id;
}

function apply(id, kind, qv) {
  var q = new Float32Array(complexInterleaved(qv, "q"));
  return complexRow(gpu.call("fmm3dbie_gpu", "points_apply", { id: id, kind: kind, q: q }).pot);
}

function free(id) {
  gpu.call("fmm3dbie_gpu", "free", { id: id });
}

function complexRow(pot) {
  var n = pot.length / 2;
  var re = new Float64Array(n), im = new Float64Array(n);
  for (var i = 0; i < n; i++) { re[i] = pot[2 * i]; im[i] = pot[2 * i + 1]; }
  return RTV.tensor(re, [1, n], im);
}

var SURFACE_KERNELS = { sprime: 0, s_kappa: 1, sprime_kappa: 2, dprime_diff: 3, s: 4, d: 5 };

function surfaceCreate(args, nargout) {
  var sv = args[2];
  if (!sv || sv.kind !== "tensor" || sv.shape[0] !== 12 || sv.imag)
    throw new RuntimeError("fmm3dbie_gpu: srcvals must be a real 12-by-npts array");
  var r = gpu.call("fmm3dbie_gpu", "surface_create", {
    norder: realScalar(args[1], "norder"),
    srcvals: new Float64Array(sv.data),
    zk: realScalar(args[3], "zk"),
    alpha: realScalar(args[4], "alpha"),
    lam: realScalar(args[5], "lam"),
    ifinout: realScalar(args[6], "ifinout"),
  });
  if (nargout < 2) return r.id;
  var info = {};
  Object.keys(r.info).forEach(function (k) { info[k] = RTV.num(r.info[k]); });
  return [r.id, RTV.struct(info)];
}

function surfaceApply(args) {
  var name = str(args[2], "kernel");
  var m = SURFACE_KERNELS[name];
  if (m === undefined) throw new RuntimeError("fmm3dbie_gpu: unknown kernel '" + name + "'");
  var q = complexInterleaved(args[3], "sigma");
  return complexRow(gpu.call("fmm3dbie_gpu", "surface_apply", { id: realScalar(args[1], "h"), m: m, q: q }).pot);
}

function complexColumn(v) {
  var t = complexRow(v);
  return RTV.tensor(t.data, [t.data.length, 1], t.imag);
}

// GMRES runs here, in numbl's worker, applying the operator through GPU
// passes; the far field comes from the GPU's dense sum or from an FMM
// (_fmm3dbie_gpu_solve.numbl.js).
function surfaceSolve(args, nargout) {
  var id = realScalar(args[1], "h");
  var rhs = complexInterleaved(args[2], "rhs");
  var eps = realScalar(args[3], "eps_gmres");
  var maxit = realScalar(args[4], "maxit");
  var lam = realScalar(args[5], "lam");
  var far = args.length > 6 ? str(args[6], "far") : "dense";
  if (far !== "dense" && far !== "fmm") throw new RuntimeError("fmm3dbie_gpu: far must be 'dense' or 'fmm'");
  if (far === "fmm" && args.length < 8) throw new RuntimeError("fmm3dbie_gpu: far = 'fmm' needs the hfmm3d handle");
  var info = gpu.call("fmm3dbie_gpu", "surface_info", { id: id });
  if (rhs.length !== info.wts.length * 2)
    throw new RuntimeError("fmm3dbie_gpu: rhs must have " + info.wts.length + " entries");
  var geo = far === "fmm" ? gpu.call("fmm3dbie_gpu", "surface_farGeometry", { id: id }) : null;
  var K = importJS("_fmm3dbie_gpu_solve").makeOperator(id, info, geo, far, far === "fmm" ? args[7] : null, Math.min(1e-7, eps / 10));
  var r = importJS("_fmm3dbie_gpu_gmres").gmres(function (x) { return K(lam, x); }, rhs, info.wts, info.zid, eps, maxit);
  var phi1 = gpu.call("fmm3dbie_gpu", "surface_sKappa", { id: id, sigma: r.x }).phi1;
  var out = [complexRow(r.x), complexRow(phi1)];
  if (nargout > 2) out.push(RTV.tensor(new Float64Array(r.residuals), [r.residuals.length, 1]));
  if (nargout > 3) out.push(RTV.num(r.rres));
  return nargout <= 1 ? out[0] : out.slice(0, Math.max(nargout, 1));
}

function surfaceEval(args) {
  var r = gpu.call("fmm3dbie_gpu", "surface_eval", {
    id: realScalar(args[1], "h"),
    targ: new Float64Array(points(args[2], "targ")),
    sigma: complexInterleaved(args[3], "sigma"),
    phi1: complexInterleaved(args[4], "phi1"),
  });
  return complexColumn(r.u);
}

function nargs(args, n, usage) {
  if (args.length !== n) throw new RuntimeError("usage: " + usage);
}

register({
  resolve: function (argTypes, nargout) {
    var outs = [];
    for (var i = 0; i < Math.max(1, nargout); i++) outs.push({ kind: "unknown" });
    return {
      outputTypes: outs,
      apply: function (args, nargout) {
        if (!gpu) throw new RuntimeError("fmm3dbie_gpu: WebGPU is not available (browser only)");
        var cmd = args.length > 0 ? str(args[0], "command") : "";
        switch (cmd) {
          case "create": {
            nargs(args, 5, "h = fmm3dbie_gpu('create', zk, src, srcn, targ)");
            var src = points(args[2], "src");
            var nrm = points(args[3], "srcn");
            if (nrm.length !== src.length) throw new RuntimeError("fmm3dbie_gpu: src and srcn sizes differ");
            return create(realScalar(args[1], "zk"), src, nrm, points(args[4], "targ"));
          }
          case "apply":
            nargs(args, 4, "pot = fmm3dbie_gpu('apply', h, kind, q)");
            return apply(realScalar(args[1], "h"), str(args[2], "kind"), args[3]);
          case "free":
            nargs(args, 2, "fmm3dbie_gpu('free', h)");
            free(realScalar(args[1], "h"));
            return RTV.tensor(new Float64Array(0), [0, 0]);
          case "freeall":
            gpu.call("fmm3dbie_gpu", "freeall", {});
            return RTV.tensor(new Float64Array(0), [0, 0]);
          case "count":
            return gpu.call("fmm3dbie_gpu", "count", {});
          case "surface_create":
            nargs(args, 7, "h = fmm3dbie_gpu('surface_create', norder, srcvals, zk, alpha, lam, ifinout)");
            return surfaceCreate(args, nargout);
          case "surface_apply":
            nargs(args, 4, "v = fmm3dbie_gpu('surface_apply', h, kernel, sigma)");
            return surfaceApply(args);
          case "surface_solve":
            if (args.length !== 6 && args.length !== 7 && args.length !== 8)
              throw new RuntimeError("usage: [sigma, phi1, errs, rres] = fmm3dbie_gpu('surface_solve', h, rhs, eps_gmres, maxit, lam[, far, hfmm])");
            return surfaceSolve(args, nargout);
          case "surface_eval":
            nargs(args, 5, "u = fmm3dbie_gpu('surface_eval', h, targ, sigma, phi1)");
            return surfaceEval(args);
          case "surface_compare_far": {
            // testing: [d0 d1 d2 d4 d23 d5] = fmm3dbie_gpu('surface_compare_far', h, x, phi1, phi2, @hfmm3d)
            var cid = realScalar(args[1], "h");
            var cinfo = gpu.call("fmm3dbie_gpu", "surface_info", { id: cid });
            var cgeo = gpu.call("fmm3dbie_gpu", "surface_farGeometry", { id: cid });
            var d = importJS("_fmm3dbie_gpu_solve").comparePasses(cid, cinfo, cgeo, args[5], args.length > 6 ? realScalar(args[6], "eps") : 1e-9,
              complexInterleaved(args[2], "x"), complexInterleaved(args[3], "phi1"), complexInterleaved(args[4], "phi2"));
            return RTV.tensor(new Float64Array(d), [1, d.length]);
          }
          case "h3d_slp": {
            nargs(args, 5, "pot = fmm3dbie_gpu('h3d_slp', zk, src, q, targ)");
            var s = points(args[2], "src");
            var id = create(realScalar(args[1], "zk"), s, null, points(args[4], "targ"));
            try {
              return apply(id, "slp", args[3]);
            } finally {
              free(id);
            }
          }
        }
        throw new RuntimeError("fmm3dbie_gpu: unknown command '" + cmd + "'");
      },
    };
  },
});
