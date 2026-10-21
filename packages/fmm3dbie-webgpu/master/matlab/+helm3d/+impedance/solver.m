function [densities, varargout] = solver(S, zlams, rhs, eps, zk, alpha, opts)
% HELM3D.IMPEDANCE.SOLVER  fmm3dbie's impedance solver on the GPU (numbl).
%
% [densities, errs, rres, Q] = helm3d.impedance.solver(S, zlams, rhs, eps, zk, alpha, opts)
%
% Same signature and equation as fmm3dbie: solves du/dn + 1i*zk*zlams*u = rhs
% with u = S_k[sigma] + 1i*alpha*D_k[S_{i|k|}[sigma]], and returns
% densities = [sigma; S_{i|k|}[sigma]] (2, npts), the GMRES relative
% residuals errs and the final residual rres. opts.eps_gmres (default eps),
% opts.maxit (200) and opts.ifout (1: exterior, 0: interior) as in fmm3dbie.
%
% Differences from fmm3dbie:
%  - quad patches (iptype 11) of a single order; zk real; zlams a real scalar
%  - single precision on the GPU, so eps_gmres below 1e-6 is raised to 1e-6
%  - opts.far_field: 'dense' (default) sums the far field directly on the GPU,
%    O(N^2); 'fmm' uses hfmm3d from the fmm3d package (single-threaded
%    WebAssembly, four calls per GMRES iteration), O(N). Both give the same
%    accuracy; on the GPUs we tried, 'dense' is faster at every size whose
%    near-field weights fit in GPU memory. helm3d.impedance.eval always uses
%    the dense sum.
%  - Q holds the GPU state (Q.gpu_handle) instead of quadrature weights. Pass
%    it as opts.quadrature_correction to a later solve (with the same S, zk,
%    alpha and ifout) or as opts.precomp_quadrature to helm3d.impedance.eval,
%    to skip the near-quadrature setup. Free it with
%    fmm3dbie_gpu('free', Q.gpu_handle). When Q is not requested, the GPU
%    state is freed before returning.
  if nargin < 7, opts = []; end
  eps_gmres = eps;
  if isfield(opts, 'eps_gmres'), eps_gmres = opts.eps_gmres; end
  maxit = 200;
  if isfield(opts, 'maxit'), maxit = opts.maxit; end
  ifout = 1;
  if isfield(opts, 'ifout'), ifout = opts.ifout; end
  far_field = 'dense';
  if isfield(opts, 'far_field'), far_field = opts.far_field; end
  if ~any(strcmp(far_field, {'dense', 'fmm'}))
    error('helm3d.impedance.solver: opts.far_field must be ''dense'' or ''fmm''');
  end

  norder = check_surface(S);
  if ~isreal(zk) || ~isscalar(zk)
    error('helm3d.impedance.solver: zk must be a real scalar on the GPU');
  end
  if ~isreal(zlams) || any(zlams(:) ~= zlams(1))
    error('helm3d.impedance.solver: zlams must be a real constant on the GPU');
  end
  lam = zlams(1);
  if numel(rhs) ~= S.npts
    error('helm3d.impedance.solver: rhs must have npts entries');
  end
  if eps_gmres < 1e-6
    warning('helm3d.impedance.solver: eps_gmres raised to 1e-6 (single precision on the GPU)');
    eps_gmres = 1e-6;
  end

  reuse = isfield(opts, 'quadrature_correction') && isfield(opts.quadrature_correction, 'gpu_handle');
  if reuse
    Q = opts.quadrature_correction;
    if Q.zk ~= zk || Q.alpha ~= alpha || Q.ifout ~= ifout || Q.npts ~= S.npts
      error('helm3d.impedance.solver: quadrature_correction was made for other parameters');
    end
    h = Q.gpu_handle;
  else
    h = fmm3dbie_gpu('surface_create', norder, extract_arrays(S), zk, alpha, lam, ifout);
    Q = struct('gpu_handle', h, 'zk', zk, 'alpha', alpha, 'ifout', ifout, 'npts', S.npts);
  end
  try
    if strcmp(far_field, 'fmm')
      [sigma, phi1, errs, rres] = fmm3dbie_gpu('surface_solve', h, rhs(:), eps_gmres, maxit, lam, 'fmm', @hfmm3d);
    else
      [sigma, phi1, errs, rres] = fmm3dbie_gpu('surface_solve', h, rhs(:), eps_gmres, maxit, lam, 'dense');
    end
  catch err
    if ~reuse, fmm3dbie_gpu('free', h); end
    rethrow(err);
  end
  densities = [sigma; phi1];
  varargout{1} = errs;
  varargout{2} = rres;
  varargout{3} = Q;
  if nargout < 4 && ~reuse
    fmm3dbie_gpu('free', h);
  end
end

function norder = check_surface(S)
  if any(S.iptype ~= 11) || any(S.norders ~= S.norders(1))
    error('helm3d.impedance.solver: the GPU version needs quad patches (iptype 11) of one order');
  end
  norder = S.norders(1);
end
