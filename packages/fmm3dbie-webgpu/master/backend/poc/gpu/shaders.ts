/**
 * WGSL compute kernels for the Helmholtz layer potentials on quadrilateral
 * patch surfaces, in single precision. A port of fmm3dbie_gpu/cuda_ops.py
 * from the CUDA proof of concept, with two changes that single precision
 * requires:
 *
 *  1. For a target on the surface, the displacement y - x_t from the target to
 *     a quadrature node is formed in the coordinates of the patch chart rather
 *     than as a difference of two rounded positions, so that it carries full
 *     relative precision however small it is. Without this the double-layer
 *     kernels, whose numerators (y - x_t).n are O(r^2), lose most of their
 *     digits near the target. There are two charts: the superellipsoid face
 *     chart of the rounded cube (gmode 0), and for any other surface the
 *     Legendre interpolant of the node data (gmode 1), where the displacement
 *     comes from divided differences of the Legendre polynomials.
 *  2. The weakly singular difference D'_k - D'_kappa is evaluated from a short
 *     power series when k r < 1, where the direct formula subtracts two nearly
 *     equal O(1/r^3) terms.
 *
 * Six kernels are handled together, in the order used by fmm3dbie's
 * helm_rpcomb_imp solver:
 *
 *     0: S'_k   1: S_kappa   2: S'_kappa   3: D'_k - D'_kappa   4: S_k   5: D_k
 *
 * with kappa = i|k| and k real, G_k = exp(ikr)/(4 pi r), S' the normal
 * derivative at the target, D the normal derivative at the source.
 */

export const NT = 64; // threads per workgroup in near_weights
export const NORD1_MAX = 11; // norder <= 10
export const NPOLS_MAX = 121;
export const DENSE_WG = 128;

const COMMON = /* wgsl */ `
const OVER4PI: f32 = 0.07957747154594767;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// (cos x, sin x) to a few ulp. WGSL only asks sin and cos for an absolute error
// of 2^-11 on [-pi, pi], and some implementations (SwiftShader among them) use
// approximations that coarse, so we reduce by pi/2 in three parts (Cody-Waite)
// and use the minimax polynomials of Cephes on [-pi/4, pi/4].
fn sincos(x: f32) -> vec2f {
  let q = round(x * 0.63661977236758134);
  var r = x - q * 1.5703125;
  r = r - q * 4.837512969970703125e-4;
  r = r - q * 7.54978995489188216e-8;
  let z = r * r;
  let s = r + r * z * (-1.6666654611e-1 + z * (8.3321608736e-3 + z * (-1.9515295891e-4)));
  let c = 1.0 - 0.5 * z + z * z * (4.166664568298827e-2 + z * (-1.388731625493765e-3 + z * 2.443315711809948e-5));
  let iq = i32(q) & 3;
  switch (iq) {
    case 0: { return vec2f(c, s); }
    case 1: { return vec2f(-s, c); }
    case 2: { return vec2f(-c, -s); }
    default: { return vec2f(s, -c); }
  }
}

fn ipow(x: f32, n: i32) -> f32 {
  var r = 1.0;
  for (var i = 0; i < n; i++) { r *= x; }
  return r;
}

// sum_{j=0}^{m} c^j d^{m-j}, so that c^{m+1} - d^{m+1} = (c - d) * geomsum(c, d, m)
fn geomsum(c: f32, d: f32, m: i32) -> f32 {
  var acc = 1.0;
  var dm = 1.0;
  for (var j = 1; j <= m; j++) { dm *= d; acc = c * acc + dm; }
  return acc;
}

// The six kernels at one (target, source) pair, times a real weight w.
// dx = x_t - y. K[3] is only computed when need3 is set.
fn kernels6(dx: vec3f, nt: vec3f, ns: vec3f, w: f32, k: f32, need3: bool) -> array<vec2f, 6> {
  let r2 = dot(dx, dx);
  let r = sqrt(r2);
  let rinv = 1.0 / r;
  let r2inv = rinv * rinv;
  let dnt = dot(dx, nt);
  let dns = dot(dx, ns);
  let kr = k * r;
  let E0 = OVER4PI * rinv * w;
  let ek = sincos(kr);
  let ec = exp(-kr);
  let Ek = E0 * ek;
  let Ec = E0 * ec;
  let fk = vec2f(1.0, -kr);
  let fc = 1.0 + kr;
  let fkEk = cmul(fk, Ek);
  var K: array<vec2f, 6>;
  K[4] = Ek;
  K[1] = vec2f(Ec, 0.0);
  K[0] = -dnt * r2inv * fkEk;
  K[2] = vec2f(-dnt * fc * Ec * r2inv, 0.0);
  K[5] = dns * r2inv * fkEk;
  K[3] = vec2f(0.0, 0.0);
  if (need3) {
    let nn = dot(ns, nt);
    var A: vec2f;  // ((1 - ikr) e^{ikr} - (1 + kr) e^{-kr})
    var B: vec2f;  // ((3 - 3ikr - k^2r^2) e^{ikr} - (3 + 3kr + k^2r^2) e^{-kr})
    if (kr < 1.0) {
      // both are sum_{n>=2} b_n (kr)^n (i^n - (-1)^n), with b_n = (1-n)/n! and (n-1)(n-3)/n!
      A = vec2f(0.0, 0.0);
      B = vec2f(0.0, 0.0);
      var t = 0.5 * kr * kr;
      for (var n = 2; n <= 14; n++) {
        var cn: vec2f;
        switch (n % 4) {
          case 0: { cn = vec2f(0.0, 0.0); }
          case 1: { cn = vec2f(1.0, 1.0); }
          case 2: { cn = vec2f(-2.0, 0.0); }
          default: { cn = vec2f(1.0, -1.0); }
        }
        let nf = f32(n);
        A += ((1.0 - nf) * t) * cn;
        B += ((nf - 1.0) * (nf - 3.0) * t) * cn;
        t *= kr / (nf + 1.0);
      }
    } else {
      A = cmul(fk, ek) - vec2f(fc * ec, 0.0);
      let gk = vec2f(3.0 - kr * kr, -3.0 * kr);
      let gc = 3.0 + 3.0 * kr + kr * kr;
      B = cmul(gk, ek) - vec2f(gc * ec, 0.0);
    }
    K[3] = E0 * (nn * r2inv * A - dns * dnt * r2inv * r2inv * B);
  }
  return K;
}

fn legendre(x: f32, norder: i32, P: ptr<function, array<f32, ${NORD1_MAX}>>) {
  (*P)[0] = 1.0;
  if (norder >= 1) { (*P)[1] = x; }
  for (var n = 1; n < norder; n++) {
    (*P)[n + 1] = ((2.0 * f32(n) + 1.0) * x * (*P)[n] - f32(n) * (*P)[n - 1]) / f32(n + 1);
  }
}
`;

/** The patch charts; they need a gpar binding. */
const CHART = /* wgsl */ `
struct Geo { x: vec3f, ru: vec3f, rv: vec3f, d: vec3f }

// xyz, r_u, r_v of the superellipsoid face chart of patch ip at (u, v), and the
// displacement d = y - x_t. With rel set, x_t is the chart point at (ut, vt) and d
// is formed in chart coordinates; otherwise d = y - xt for a given xt.
// gpar: 4 vec4 per patch, [n, s0], [e1, s1], [e2, t0], [0, t1].
fn chart_rel(ip: i32, u: f32, v: f32, ut: f32, vt: f32, rel: bool, xt: vec3f, ga: f32, gpe: i32) -> Geo {
  let g0 = gpar[4 * ip]; let g1 = gpar[4 * ip + 1]; let g2 = gpar[4 * ip + 2]; let g3 = gpar[4 * ip + 3];
  let nf = g0.xyz; let e1 = g1.xyz; let e2 = g2.xyz;
  let hs = 0.5 * (g1.w - g0.w); let ht = 0.5 * (g3.w - g2.w);
  let s = g0.w + (u + 1.0) * hs; let t = g2.w + (v + 1.0) * ht;
  let c = nf + s * e1 + t * e2;
  let S = ipow(c.x, gpe) + ipow(c.y, gpe) + ipow(c.z, gpe);
  let N = pow(S, 1.0 / f32(gpe));
  let gN = vec3f(ipow(c.x / N, gpe - 1), ipow(c.y / N, gpe - 1), ipow(c.z / N, gpe - 1));
  let dNu = dot(gN, e1); let dNv = dot(gN, e2);
  var out: Geo;
  out.x = ga * c / N;
  out.ru = ga * hs * (e1 / N - c * dNu / (N * N));
  out.rv = ga * ht * (e2 / N - c * dNv / (N * N));
  if (rel) {
    let st = g0.w + (ut + 1.0) * hs; let tt = g2.w + (vt + 1.0) * ht;
    let ct = nf + st * e1 + tt * e2;
    let dc = ((u - ut) * hs) * e1 + ((v - vt) * ht) * e2;
    let St = ipow(ct.x, gpe) + ipow(ct.y, gpe) + ipow(ct.z, gpe);
    let Nt = pow(St, 1.0 / f32(gpe));
    // N^p - Nt^p, then N - Nt, both without cancellation
    let dP = dc.x * geomsum(c.x, ct.x, gpe - 1) + dc.y * geomsum(c.y, ct.y, gpe - 1) + dc.z * geomsum(c.z, ct.z, gpe - 1);
    let dN = dP / geomsum(N, Nt, gpe - 1);
    out.d = ga * (dc * Nt - ct * dN) / (N * Nt);
  } else {
    out.d = out.x - xt;
  }
  return out;
}

// P_n(x) - P_n(xt) = (x - xt) * dd[n], from the three-term recurrence:
// dd[n+1] = ((2n+1) (P_n(xt) + x dd[n]) - n dd[n-1]) / (n+1).
fn legendre_dd(x: f32, norder: i32, Pt: ptr<function, array<f32, ${NORD1_MAX}>>, dd: ptr<function, array<f32, ${NORD1_MAX}>>) {
  (*dd)[0] = 0.0;
  if (norder >= 1) { (*dd)[1] = 1.0; }
  for (var n = 1; n < norder; n++) {
    (*dd)[n + 1] = ((2.0 * f32(n) + 1.0) * ((*Pt)[n] + x * (*dd)[n]) - f32(n) * (*dd)[n - 1]) / f32(n + 1);
  }
}

fn gcoef(i: u32) -> f32 {
  return gpar[i >> 2u][i & 3u];
}

// xyz, r_u, r_v of the Legendre interpolant of patch ip at (u, v), and d = y - x_t.
// gpar holds the coefficients of every patch, 9 * npols floats each (xyz, r_u, r_v).
// With rel set, d = x(u, v) - x(ut, vt) + e is formed with divided differences, where
// e = x(ut, vt) - x_t is the (small) gap between the target and the interpolant at
// its preimage, computed on the CPU; otherwise d = y - xt.
fn chart_poly(ip: i32, u: f32, v: f32, ut: f32, vt: f32, rel: bool, xt: vec3f, e: vec3f, norder: i32) -> Geo {
  let n1 = norder + 1;
  let npols = n1 * n1;
  let base = u32(ip) * u32(9 * npols);
  var Pu: array<f32, ${NORD1_MAX}>;
  var Pv: array<f32, ${NORD1_MAX}>;
  legendre(u, norder, &Pu);
  legendre(v, norder, &Pv);
  var out: Geo;
  out.x = vec3f(0.0); out.ru = vec3f(0.0); out.rv = vec3f(0.0);
  for (var b = 0; b < n1; b++) {
    for (var a = 0; a < n1; a++) {
      let P = Pu[a] * Pv[b];
      let n = u32(b * n1 + a);
      let np = u32(npols);
      out.x += P * vec3f(gcoef(base + n), gcoef(base + np + n), gcoef(base + 2u * np + n));
      out.ru += P * vec3f(gcoef(base + 3u * np + n), gcoef(base + 4u * np + n), gcoef(base + 5u * np + n));
      out.rv += P * vec3f(gcoef(base + 6u * np + n), gcoef(base + 7u * np + n), gcoef(base + 8u * np + n));
    }
  }
  if (rel) {
    var Put: array<f32, ${NORD1_MAX}>;
    var Pvt: array<f32, ${NORD1_MAX}>;
    var du: array<f32, ${NORD1_MAX}>;
    var dv: array<f32, ${NORD1_MAX}>;
    legendre(ut, norder, &Put);
    legendre(vt, norder, &Pvt);
    legendre_dd(u, norder, &Put, &du);
    legendre_dd(v, norder, &Pvt, &dv);
    // P_a(u) P_b(v) - P_a(ut) P_b(vt) = (u - ut) du_a P_b(v) + (v - vt) P_a(ut) dv_b
    let hu = u - ut;
    let hv = v - vt;
    var d = e;
    for (var b = 0; b < n1; b++) {
      for (var a = 0; a < n1; a++) {
        let w = hu * du[a] * Pv[b] + hv * Put[a] * dv[b];
        let n = u32(b * n1 + a);
        let np = u32(npols);
        d += w * vec3f(gcoef(base + n), gcoef(base + np + n), gcoef(base + 2u * np + n));
      }
    }
    out.d = d;
  } else {
    out.d = out.x - xt;
  }
  return out;
}

`;

/**
 * Quadtree leaves on the reference square of a near (non-self) patch, refined
 * while the target is within rfac_q times the radius of a leaf or the leaf is
 * large compared with the wavelength. Two passes: counts only (fill = 0), then
 * fill using offsets. Self pairs are skipped; their items come from the CPU.
 * Items are 2 vec4: [kind, cu, cv, h], [0, 0, 0, 0] with kind 0 for a leaf.
 */
export const BUILD_LEAVES = /* wgsl */ `
struct Params {
  n: u32,
  pair_offset: u32,
  fill: u32,
  nlmax: i32,
  rfac_q: f32,
  zk: f32,
  kR_max: f32,
  ga: f32,
  gpe: i32,
  gmode: i32,
  norder: i32,
}
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> pairs_i: array<vec2i>;
@group(0) @binding(2) var<storage, read> pairs_f: array<vec4f>;
@group(0) @binding(3) var<storage, read> targ: array<vec4f>;
@group(0) @binding(4) var<storage, read> gpar: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> counts: array<u32>;
@group(0) @binding(6) var<storage, read> offsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> items: array<vec4f>;
${COMMON}
${CHART}

@compute @workgroup_size(64)
fn build_leaves(@builtin(global_invocation_id) gid: vec3u) {
  let pl = gid.x;
  if (pl >= U.n) { return; }
  let p = pl + U.pair_offset;
  let ip = pairs_i[p].y;
  let it = pairs_i[p].x;
  let kind = i32(pairs_f[2u * p].z);
  if (kind == 2) {
    if (U.fill == 0u) { counts[pl] = 0u; }
    return;
  }
  let tx = targ[2 * it].xyz;
  var su: array<f32, 64>;
  var sv: array<f32, 64>;
  var sh: array<f32, 64>;
  var sd: array<i32, 64>;
  var sp = 1;
  su[0] = 0.0; sv[0] = 0.0; sh[0] = 1.0; sd[0] = 0;
  var n = 0u;
  let rq2 = U.rfac_q * U.rfac_q;
  let kmax2 = U.kR_max * U.kR_max;
  let k2 = U.zk * U.zk;
  while (sp > 0) {
    sp--;
    let cu = su[sp]; let cv = sv[sp]; let h = sh[sp]; let d = sd[sp];
    var refine = false;
    if (d < U.nlmax) {
      // centroid and radius from a 3x3 sample grid
      var xc: array<vec3f, 9>;
      var cen = vec3f(0.0);
      for (var q = 0; q < 9; q++) {
        let uq = cu + h * f32(q % 3 - 1);
        let vq = cv + h * f32(q / 3 - 1);
        var g: Geo;
        if (U.gmode == 1) { g = chart_poly(ip, uq, vq, 0.0, 0.0, false, vec3f(0.0), vec3f(0.0), U.norder); }
        else { g = chart_rel(ip, uq, vq, 0.0, 0.0, false, vec3f(0.0), U.ga, U.gpe); }
        xc[q] = g.x;
        cen += g.x / 9.0;
      }
      var R2 = 0.0;
      for (var q = 0; q < 9; q++) { let e = xc[q] - cen; R2 = max(R2, dot(e, e)); }
      let e = tx - cen;
      let d2 = dot(e, e);
      refine = (d2 < rq2 * R2) || (k2 * R2 > kmax2);
    }
    if (refine && sp <= 60) {
      let hh = 0.5 * h;
      su[sp] = cu - hh; sv[sp] = cv - hh; sh[sp] = hh; sd[sp] = d + 1; sp++;
      su[sp] = cu + hh; sv[sp] = cv - hh; sh[sp] = hh; sd[sp] = d + 1; sp++;
      su[sp] = cu + hh; sv[sp] = cv + hh; sh[sp] = hh; sd[sp] = d + 1; sp++;
      su[sp] = cu - hh; sv[sp] = cv + hh; sh[sp] = hh; sd[sp] = d + 1; sp++;
    } else {
      if (U.fill != 0u) {
        let o = 2u * (offsets[pl] + n);
        items[o] = vec4f(0.0, cu, cv, h);
        items[o + 1u] = vec4f(0.0);
      }
      n++;
    }
  }
  if (U.fill == 0u) { counts[pl] = n; }
}
`;

/**
 * Near-field weights, one workgroup per (target, patch) pair. The nodes of the
 * pair's items (self-patch polar segments or quadtree leaves) are evaluated
 * ${NT} at a time; each thread then accumulates the Legendre moments
 * I_n = sum K P_n |J| w for its basis indices n, and finally the moments are
 * turned into node weights w_j = sum_n U[n, j] I_n.
 *
 * Self items are 2 vec4: [1, edge, xi0, xi1], [h, xc, heff, 0]: on edge e the
 * polar rule uses rho in [0, 1] along the ray and the sinh substitution
 * x = xc + heff sinh(xi), xi in [xi0, xi1], along the edge.
 * tables: umat (npols*npols) followed by the GL rules gl_x, gl_w (nq), glr_x,
 * glr_w (nqr, on [0, 1]) and glx_x, glx_w (nqx).
 */
export const NEAR_WEIGHTS = /* wgsl */ `
struct Params {
  n: u32,
  pair_offset: u32,
  norder: i32,
  npols: i32,
  nq: i32,
  nqr: i32,
  nqx: i32,
  zk: f32,
  ga: f32,
  gpe: i32,
  gmode: i32,
}
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> pairs_i: array<vec2i>;
@group(0) @binding(2) var<storage, read> pairs_f: array<vec4f>;
@group(0) @binding(3) var<storage, read> targ: array<vec4f>;
@group(0) @binding(4) var<storage, read> gpar: array<vec4f>;
@group(0) @binding(5) var<storage, read> item_ptr: array<u32>;
@group(0) @binding(6) var<storage, read> items: array<vec4f>;
@group(0) @binding(7) var<storage, read> tables: array<f32>;
@group(0) @binding(8) var<storage, read_write> W: array<vec2f>;
${COMMON}
${CHART}

const NT = ${NT};
var<workgroup> sK: array<vec2f, ${6 * NT}>;
var<workgroup> sLu: array<f32, ${NORD1_MAX * NT}>;
var<workgroup> sLv: array<f32, ${NORD1_MAX * NT}>;
var<workgroup> sM: array<vec2f, ${6 * NPOLS_MAX}>;
var<workgroup> sNtot: i32;

@compute @workgroup_size(${NT})
fn near_weights(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let pl = wid.x;
  let p = pl + U.pair_offset;
  let t = i32(lid.x);
  let it = pairs_i[p].x;
  let ip = pairs_i[p].y;
  let ut = pairs_f[2u * p].x;
  let vt = pairs_f[2u * p].y;
  let kind = i32(pairs_f[2u * p].z);
  let egap = pairs_f[2u * p + 1u].xyz;
  let is_self = kind == 2;
  let onsurf = kind >= 1;
  let xt = targ[2 * it].xyz;
  let nt = targ[2 * it + 1].xyz;
  let n1 = U.norder + 1;
  let npols = U.npols;
  let i0 = item_ptr[pl];
  let nodes_per_item = select(U.nq * U.nq, U.nqr * U.nqx, is_self);
  if (t == 0) { sNtot = i32(item_ptr[pl + 1u] - i0) * nodes_per_item; }
  let ntot = workgroupUniformLoad(&sNtot);
  let oq = npols * npols;               // gl tables start after umat
  let or = oq + 2 * U.nq;               // glr_x
  let ox = or + 2 * U.nqr;              // glx_x

  var Ir: array<vec2f, 12>;
  for (var a = 0; a < 12; a++) { Ir[a] = vec2f(0.0, 0.0); }

  for (var base = 0; base < ntot; base += NT) {
    let g = base + t;
    var K: array<vec2f, 6>;
    var Lu: array<f32, ${NORD1_MAX}>;
    var Lv: array<f32, ${NORD1_MAX}>;
    let live = g < ntot;
    if (live) {
      let item = g / nodes_per_item;
      let loc = g - item * nodes_per_item;
      let it0 = items[2u * (i0 + u32(item))];
      let it1 = items[2u * (i0 + u32(item)) + 1u];
      var u: f32; var v: f32; var wq: f32;
      if (is_self) {
        let e = i32(it0.y);
        let ir = loc / U.nqx;
        let ix = loc - ir * U.nqx;
        let h = it1.x; let xc = it1.y; let heff = it1.z;
        let rho = tables[or + ir];
        let wr = tables[or + U.nqr + ir];
        let xi = it0.z + 0.5 * (it0.w - it0.z) * (tables[ox + ix] + 1.0);
        let wxi = 0.5 * (it0.w - it0.z) * tables[ox + U.nqx + ix];
        let x = xc + heff * sinh(xi);
        let dx = heff * cosh(xi) * wxi;
        if (e < 2) { u = ut + rho * h; v = vt + rho * x; } else { u = ut + rho * x; v = vt + rho * h; }
        wq = abs(h) * rho * wr * dx;
      } else {
        let a = loc / U.nq;
        let b = loc - a * U.nq;
        u = it0.y + it0.w * tables[oq + a];
        v = it0.z + it0.w * tables[oq + b];
        wq = it0.w * it0.w * tables[oq + U.nq + a] * tables[oq + U.nq + b];
      }
      var G: Geo;
      if (U.gmode == 1) { G = chart_poly(ip, u, v, ut, vt, onsurf, xt, egap, U.norder); }
      else { G = chart_rel(ip, u, v, ut, vt, onsurf, xt, U.ga, U.gpe); }
      var ns = cross(G.ru, G.rv);
      let jac = length(ns);
      ns = ns / jac;
      K = kernels6(-G.d, nt, ns, wq * jac, U.zk, true);
      legendre(u, U.norder, &Lu);
      legendre(v, U.norder, &Lv);
    }
    for (var m = 0; m < 6; m++) { sK[m * NT + t] = select(vec2f(0.0, 0.0), K[m], live); }
    for (var a = 0; a < n1; a++) {
      sLu[a * NT + t] = select(0.0, Lu[a], live);
      sLv[a * NT + t] = select(0.0, Lv[a], live);
    }
    workgroupBarrier();
    for (var acc = 0; acc < 2; acc++) {
      let n = t + acc * NT;
      if (n < npols) {
        let a = n % n1;
        let b = n / n1;
        for (var i = 0; i < NT; i++) {
          let P = sLu[a * NT + i] * sLv[b * NT + i];
          for (var m = 0; m < 6; m++) { Ir[acc * 6 + m] += P * sK[m * NT + i]; }
        }
      }
    }
    workgroupBarrier();
  }

  // moments -> node weights, w_j = sum_n U[n, j] I_n
  for (var acc = 0; acc < 2; acc++) {
    let n = t + acc * NT;
    if (n < npols) { for (var m = 0; m < 6; m++) { sM[m * ${NPOLS_MAX} + n] = Ir[acc * 6 + m]; } }
  }
  workgroupBarrier();
  for (var j = t; j < npols; j += NT) {
    for (var m = 0; m < 6; m++) {
      var w = vec2f(0.0, 0.0);
      for (var n = 0; n < npols; n++) { w += tables[n * npols + j] * sM[m * ${NPOLS_MAX} + n]; }
      W[(u32(p) * 6u + u32(m)) * u32(npols) + u32(j)] = w;
    }
  }
}
`;

/**
 * Far-field pass over the patches j0 <= j < j1: out[m][i] += sum over patches j
 * not near target i of sum_l K_m(x_i, y_l) w_l q_l over the oversampled nodes
 * of patch j, for the kernels selected in selmask (4 bits per kernel: 0 skip,
 * 1 density qa, 2 density qb). A patch is near target i when it appears in
 * the target's near list (row_ptr, pair_p, ascending), so the split between
 * this sum and the near weights is exact.
 * srcgeo: 2 vec4 per node, [x, w], [n, 0]; srcq: vec4 per node (qa, qb).
 */
export const DENSE_APPLY = /* wgsl */ `
struct Params {
  ntarg: u32,
  j0: u32,
  j1: u32,
  npols_o: u32,
  zk: f32,
  selmask: u32,
}
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> targ: array<vec4f>;
@group(0) @binding(2) var<storage, read> srcgeo: array<vec4f>;
@group(0) @binding(3) var<storage, read> srcq: array<vec4f>;
@group(0) @binding(4) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(5) var<storage, read> pair_p: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<vec2f>;
${COMMON}

const WG = ${DENSE_WG}u;
var<workgroup> tile: array<vec4f, ${3 * DENSE_WG}>;

@compute @workgroup_size(${DENSE_WG})
fn dense_apply(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let i = gid.x;
  let valid = i < U.ntarg;
  var xt = vec3f(0.0);
  var nt = vec3f(0.0);
  var cur = 0u;
  var end = 0u;
  if (valid) {
    xt = targ[2u * i].xyz;
    nt = targ[2u * i + 1u].xyz;
    cur = row_ptr[i];
    end = row_ptr[i + 1u];
    while (cur < end && pair_p[cur] < U.j0) { cur++; }
  }
  var sel: array<u32, 6>;
  for (var m = 0u; m < 6u; m++) { sel[m] = (U.selmask >> (4u * m)) & 3u; }
  let need3 = sel[3] != 0u;
  var acc: array<vec2f, 6>;
  for (var m = 0; m < 6; m++) { acc[m] = vec2f(0.0, 0.0); }
  for (var j = U.j0; j < U.j1; j++) {
    var far = valid;
    if (valid && cur < end && pair_p[cur] == j) { far = false; cur++; }
    for (var base = 0u; base < U.npols_o; base += WG) {
      let l = base + lid.x;
      if (l < U.npols_o) {
        let s = j * U.npols_o + l;
        tile[3u * lid.x] = srcgeo[2u * s];
        tile[3u * lid.x + 1u] = srcgeo[2u * s + 1u];
        tile[3u * lid.x + 2u] = srcq[s];
      }
      workgroupBarrier();
      if (far) {
        let nl = min(WG, U.npols_o - base);
        var part: array<vec2f, 6>;
        for (var m = 0; m < 6; m++) { part[m] = vec2f(0.0, 0.0); }
        for (var s = 0u; s < nl; s++) {
          let g0 = tile[3u * s];
          let g1 = tile[3u * s + 1u];
          let q = tile[3u * s + 2u];
          let K = kernels6(xt - g0.xyz, nt, g1.xyz, g0.w, U.zk, need3);
          for (var m = 0u; m < 6u; m++) {
            if (sel[m] == 1u) { part[m] += cmul(K[m], q.xy); }
            else if (sel[m] == 2u) { part[m] += cmul(K[m], q.zw); }
          }
        }
        for (var m = 0; m < 6; m++) { acc[m] += part[m]; }
      }
      workgroupBarrier();
    }
  }
  if (valid) {
    for (var m = 0u; m < 6u; m++) {
      if (sel[m] != 0u) { out[m * U.ntarg + i] += acc[m]; }
    }
  }
}
`;

/**
 * Near contributions: out[m][i] += sum over the near pairs p of target i of
 * sum_l W[p][m][l] q[d][patch nodes l], for up to four requests (m, d) packed
 * as m | d << 8 in req.
 */
export const SPARSE_APPLY = /* wgsl */ `
struct Params {
  ntarg: u32,
  npols: u32,
  npts: u32,
  nreq: u32,
  req: vec4u,
}
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> pair_p: array<u32>;
@group(0) @binding(3) var<storage, read> W: array<vec2f>;
@group(0) @binding(4) var<storage, read> q: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> out: array<vec2f>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(128)
fn sparse_apply(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.ntarg) { return; }
  var acc: array<vec2f, 4>;
  for (var r = 0u; r < 4u; r++) { acc[r] = vec2f(0.0, 0.0); }
  for (var p = row_ptr[i]; p < row_ptr[i + 1u]; p++) {
    let j = pair_p[p];
    for (var r = 0u; r < U.nreq; r++) {
      let m = U.req[r] & 0xffu;
      let d = U.req[r] >> 8u;
      let wbase = (p * 6u + m) * U.npols;
      let qbase = d * U.npts + j * U.npols;
      var s = vec2f(0.0, 0.0);
      for (var l = 0u; l < U.npols; l++) { s += cmul(W[wbase + l], q[qbase + l]); }
      acc[r] += s;
    }
  }
  for (var r = 0u; r < U.nreq; r++) {
    let m = U.req[r] & 0xffu;
    out[m * U.ntarg + i] += acc[r];
  }
}
`;
