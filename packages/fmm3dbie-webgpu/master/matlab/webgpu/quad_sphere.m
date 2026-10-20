function S = quad_sphere(a, na, c0, norder)
% QUAD_SPHERE Sphere of radius a centered at c0 as a surfer of quad patches.
%
% S = quad_sphere(a, na, c0, norder) uses an equiangular cube-sphere: each of
%  the 6 cube faces is split into na x na patches of type 11 (tensor
%  Gauss-Legendre, order norder). Pure MATLAB, unlike geometries.sphere,
%  which calls the fmm3dbie mex.
  if nargin < 1 || isempty(a), a = 1; end
  if nargin < 2 || isempty(na), na = 2; end
  if nargin < 3 || isempty(c0), c0 = [0; 0; 0]; end
  if nargin < 4 || isempty(norder), norder = 6; end
  c0 = c0(:);

  uv = polytens.lege.nodes(norder);
  npols = size(uv, 2);
  npatches = 6*na*na;
  srcvals = zeros(12, npatches*npols);

  % (e_s, e_t, e_n) per face with e_s x e_t = e_n (outward)
  E = eye(3);
  faces = {[2 3 1 1], [3 2 1 -1], [3 1 2 1], [1 3 2 -1], [1 2 3 1], [2 1 3 -1]};
  h = 2/na;
  ip = 0;
  for f = 1:6
    fc = faces{f};
    es = E(:, fc(1)); et = E(:, fc(2)); en = fc(4)*E(:, fc(3));
    for i = 1:na
      for j = 1:na
        xi = -1 + (i - 1)*h + (uv(1, :) + 1)*h/2;
        eta = -1 + (j - 1)*h + (uv(2, :) + 1)*h/2;
        s = tan(pi/4*xi);
        t = tan(pi/4*eta);
        ds = (pi/4)*(h/2)*(1 + s.^2);
        dt = (pi/4)*(h/2)*(1 + t.^2);
        c = es*s + et*t + en;
        rc = sqrt(sum(c.^2, 1));
        p = c./rc;
        % d(c/|c|) = (I - p p') dc / |c|
        dcu = es*ds;
        dcv = et*dt;
        du = a*(dcu - p.*sum(p.*dcu, 1))./rc;
        dv = a*(dcv - p.*sum(p.*dcv, 1))./rc;
        iind = ip*npols + (1:npols);
        srcvals(1:3, iind) = c0 + a*p;
        srcvals(4:6, iind) = du;
        srcvals(7:9, iind) = dv;
        srcvals(10:12, iind) = p;
        ip = ip + 1;
      end
    end
  end
  S = surfer(npatches, norder, srcvals, 11);
end
