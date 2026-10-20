% Test for fmm3dbie-webgpu. Always checks the parts that run on the CPU: the
% upstream surfer class on a quad sphere, the stand-in for the fmm3dbie MEX,
% and the GPU gateway failing cleanly without WebGPU (as in numbl on Node).
% With WebGPU (numbl in a browser), it also solves an exterior Neumann problem
% whose solution is the field of a point source inside the sphere.

a = 1.2;
S = quad_sphere(a, 2, [0; 0; 0], 4);
assert(S.npatches == 24 && S.npts == 24*25);
assert(abs(area(S) - 4*pi*a^2)/(4*pi*a^2) < 1e-8);

% Upstream routines that need the MEX fail with a clear message
try
  geometries.sphere(1);
  error('expected geometries.sphere to fail without the MEX');
catch err
  assert(contains(err.message, 'fmm3dbie-webgpu'), err.message);
end

has_gpu = true;
try
  fmm3dbie_gpu('count');
catch err
  has_gpu = false;
  assert(contains(err.message, 'WebGPU is not available'), err.message);
end

if has_gpu
  zk = 2.5;
  x0 = [0.2; -0.3; 0.1];
  G = @(x) exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0));
  dGdn = @(x, n) sum((x - x0).*n, 1).*(1i*zk*vecnorm(x - x0) - 1) ...
    .*exp(1i*zk*vecnorm(x - x0))./(4*pi*vecnorm(x - x0).^3);
  [dens, errs, ~, Q] = helm3d.neumann.solver(S, dGdn(S.r, S.n), 1e-6, zk, 1);
  targ = [2 0 0; 0 -2.5 0; 1 1 1.5].';
  p = helm3d.neumann.eval(S, dens, targ, 1e-6, zk, 1, struct('precomp_quadrature', Q));
  fmm3dbie_gpu('free', Q.gpu_handle);
  err = max(abs(p.' - G(targ)))/max(abs(G(targ)));
  fprintf('GPU Neumann self-test: %d GMRES iterations, rel err %.1e\n', numel(errs), err);
  assert(err < 1e-3);
else
  disp('WebGPU not available: skipped the GPU solve');
end

disp('SUCCESS');
