% Plane wave scattering from a sound-hard sphere, solved on the GPU through
% fmm3dbie's helm3d.neumann.solver / eval signatures (numbl + WebGPU).
% The scattered field u_s satisfies du_s/dn = -du_inc/dn on the sphere; we
% plot the real part of the total field u_inc + u_s on the plane z = 0 and
% compare u_s with the exact series solution (mie_sound_hard.m).
%
% mip install --channel mip-org/labs fmm3dbie-webgpu
% mip load fmm3dbie-webgpu

a = 1;          % sphere radius
zk = 6;         % wavenumber (ka = 6)
norder = 6;     % patch order
S = quad_sphere(a, 3, [0; 0; 0], norder);
fprintf('sphere: %d patches of order %d, %d points\n', S.npatches, norder, S.npts);

d = [1; 0; 0];  % direction of incidence
uinc = @(x) exp(1i*zk*(d.'*x));
rhs = -1i*zk*(d.'*S.n).*uinc(S.r);

t = tic;
[dens, errs, ~, Q] = helm3d.neumann.solver(S, rhs, 1e-6, zk, 1);
fprintf('solve: %d GMRES iterations, %.1f s (includes the near-quadrature setup)\n', numel(errs), toc(t));

% Field on a grid in the plane z = 0, outside the sphere
n = 121;
L = 3;
[X, Y] = meshgrid(linspace(-L, L, n));
targ = [X(:).'; Y(:).'; zeros(1, n^2)];
out = vecnorm(targ) > 1.01*a;
t = tic;
us = zeros(n^2, 1);
us(out) = helm3d.neumann.eval(S, dens, targ(:, out), 1e-6, zk, 1, struct('precomp_quadrature', Q));
fprintf('eval at %d points: %.1f s\n', nnz(out), toc(t));
fmm3dbie_gpu('free', Q.gpu_handle);

% Compare with the exact (Mie series) solution
uex = mie_sound_hard(zk, a, targ(:, out)).';
fprintf('max error vs exact solution: %.1e (relative to max |u_s| = %.2f)\n', ...
  max(abs(us(out) - uex))/max(abs(uex)), max(abs(uex)));

u = uinc(targ).' + us;
u(~out) = NaN;
figure;
imagesc(linspace(-L, L, n), linspace(-L, L, n), reshape(real(u), n, n));
axis xy equal tight;
colorbar;
hold on;
th = linspace(0, 2*pi, 200);
plot(a*cos(th), a*sin(th), 'k', 'LineWidth', 1.5);
hold off;
title(sprintf('Re u, plane wave on a sound-hard sphere, ka = %g', zk*a));
xlabel('x');
ylabel('y');
