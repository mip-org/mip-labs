function p = eval(S, densities, targinfo, eps, zk, alpha, opts)
% HELM3D.NEUMANN.EVAL  fmm3dbie's Neumann evaluator on the GPU (numbl).
%
% p = helm3d.neumann.eval(S, densities, targinfo, eps, zk, alpha, opts)
%
% The representation is that of the impedance problem, so this calls
% helm3d.impedance.eval; see there for the differences from fmm3dbie.
  if nargin < 7, opts = []; end
  p = helm3d.impedance.eval(S, densities, targinfo, eps, zk, alpha, opts);
end
