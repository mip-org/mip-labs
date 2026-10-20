function [densities, varargout] = solver(S, rhs, eps, zk, alpha, opts)
% HELM3D.NEUMANN.SOLVER  fmm3dbie's Neumann solver on the GPU (numbl).
%
% [densities, errs, rres, Q] = helm3d.neumann.solver(S, rhs, eps, zk, alpha, opts)
%
% Same signature as fmm3dbie: du/dn = rhs with u = S_k[sigma] +
% 1i*alpha*D_k[S_{i|k|}[sigma]]. This is the impedance equation with
% zlams = 0, so it calls helm3d.impedance.solver; see there for the
% differences from fmm3dbie and for Q.
  if nargin < 6, opts = []; end
  [densities, errs, rres, Q] = helm3d.impedance.solver(S, 0, rhs, eps, zk, alpha, opts);
  varargout{1} = errs;
  varargout{2} = rres;
  varargout{3} = Q;
  if nargout < 4 && ~(isfield(opts, 'quadrature_correction') && isfield(opts.quadrature_correction, 'gpu_handle'))
    fmm3dbie_gpu('free', Q.gpu_handle);
  end
end
