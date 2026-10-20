function p = eval(S, densities, targinfo, eps, zk, alpha, opts)
% HELM3D.IMPEDANCE.EVAL  fmm3dbie's impedance evaluator on the GPU (numbl).
%
% p = helm3d.impedance.eval(S, densities, targinfo, eps, zk, alpha, opts)
%
% Same signature as fmm3dbie: p = S_k[sigma] + 1i*alpha*D_k[phi1] at the
% targets targinfo.r (or a 3-by-n array), with densities = [sigma; phi1] from
% helm3d.impedance.solver; p is (ntarg, 1).
%
% Differences from fmm3dbie: off-surface targets only; quad patches (iptype
% 11) of a single order; zk real; single precision. opts.precomp_quadrature
% may be the Q returned by helm3d.impedance.solver, which reuses its GPU
% state; otherwise the GPU state is set up here (near quadrature for the
% surface nodes included, which is the slow part) and freed.
  if nargin < 7, opts = []; end
  if isstruct(targinfo) || isa(targinfo, 'surfer')
    targ = targinfo.r;
    if isfield(targinfo, 'patch_id') && any(targinfo.patch_id > 0)
      error('helm3d.impedance.eval: on-surface targets are not supported on the GPU');
    end
  else
    targ = targinfo;
  end
  if size(densities, 1) ~= 2 || size(densities, 2) ~= S.npts
    error('helm3d.impedance.eval: densities must be 2-by-npts');
  end
  if ~isreal(zk), error('helm3d.impedance.eval: zk must be real on the GPU'); end

  reuse = isfield(opts, 'precomp_quadrature') && isfield(opts.precomp_quadrature, 'gpu_handle');
  if reuse
    Q = opts.precomp_quadrature;
    if Q.zk ~= zk || Q.alpha ~= alpha || Q.npts ~= S.npts
      error('helm3d.impedance.eval: precomp_quadrature was made for other parameters');
    end
    h = Q.gpu_handle;
  else
    if any(S.iptype ~= 11) || any(S.norders ~= S.norders(1))
      error('helm3d.impedance.eval: the GPU version needs quad patches (iptype 11) of one order');
    end
    h = fmm3dbie_gpu('surface_create', S.norders(1), extract_arrays(S), zk, alpha, 0, 1);
  end
  try
    p = fmm3dbie_gpu('surface_eval', h, targ, densities(1, :), densities(2, :));
  catch err
    if ~reuse, fmm3dbie_gpu('free', h); end
    rethrow(err);
  end
  if ~reuse, fmm3dbie_gpu('free', h); end
end
