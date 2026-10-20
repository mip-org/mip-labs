/** Gauss-Legendre nodes (ascending) and weights on [-1, 1], by Newton iteration. */
export function leggauss(n: number): { x: Float64Array; w: Float64Array } {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  const m = (n + 1) >> 1;
  for (let i = 0; i < m; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let pp = 0;
    for (let iter = 0; iter < 100; iter++) {
      let p1 = 1;
      let p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1);
      const z1 = z;
      z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-15) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * pp * pp);
  }
  return { x, w };
}

/** P_0(x), ..., P_norder(x) for each x: array of shape (norder+1, n), index k*n + i. */
export function legendreVander(x: ArrayLike<number>, norder: number): Float64Array {
  const n = x.length;
  const P = new Float64Array((norder + 1) * n);
  for (let i = 0; i < n; i++) {
    P[i] = 1;
    if (norder >= 1) P[n + i] = x[i];
    for (let k = 1; k < norder; k++) {
      P[(k + 1) * n + i] = ((2 * k + 1) * x[i] * P[k * n + i] - k * P[(k - 1) * n + i]) / (k + 1);
    }
  }
  return P;
}
