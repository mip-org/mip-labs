function test_treeweave()
% Channel test for the treeweave MATLAB binding.
%
% Runs after `mip load treeweave`, using only the public API (the treeweave
% classdef). It constructs and evaluates an approximator, which drives the single
% shipped MEX (treeweave_mex) through the mwrap-generated tw_* stubs — so the
% MEX-coverage gate (built == loaded) is satisfied. Errors via assert/error on
% any failure and prints SUCCESS at the end.

rng('default');

% ---- 1-D scalar fit: exp(0.5*x) + sin(3*x) on [0, 1] ----------------------
fprintf('Testing 1-D scalar fit...\n');
f1  = @(x) exp(0.5*x(1)) + sin(3*x(1));
o1  = treeweave(f1, 0, 1, 1e-8, 'dim', 1, 'out_dim', 1);

% Stay inside [a, b) for the accuracy sweep.
X   = linspace(0, 1, 201)';
X(end) = [];
Y   = o1.eval(X);
Yref = exp(0.5*X) + sin(3*X);
err1 = max(abs(Y(:) - Yref(:)));
assert(err1 < 1e-5, 'treeweave:test', '1-D accuracy too low (max err = %.3e)', err1);

% subsref syntax o1(X) must match eval.
assert(max(abs(o1(X) - Y)) < 1e-12, 'treeweave:test', 'subsref obj(X) disagrees with eval');

% Out-of-domain returns NaN.
assert(isnan(o1.eval(-5)), 'treeweave:test', 'out-of-domain did not return NaN');
delete(o1);

% ---- 2-D -> 3-D vector fit ------------------------------------------------
fprintf('Testing 2-D -> 3-D vector fit...\n');
f2 = @(x) [sin(x(1)+x(2)); cos(x(1)-x(2)); x(1)*x(2)];
o2 = treeweave(f2, [-1, -1], [1, 1], 1e-6, 'dim', 2, 'max_memory_mib', 64);
assert(o2.output_dim() == 3, 'treeweave:test', 'out_dim inference failed (expected 3)');

[gx, gy] = meshgrid(linspace(-1, 1, 50), linspace(-1, 1, 50));
Xg = [gx(:), gy(:)];
Yg = o2.eval(Xg);
Yr = [sin(Xg(:,1)+Xg(:,2)), cos(Xg(:,1)-Xg(:,2)), Xg(:,1).*Xg(:,2)];
err2 = max(max(abs(Yg - Yr)));
assert(err2 < 1e-4, 'treeweave:test', '2-D->3-D accuracy too low (max err = %.3e)', err2);
assert(isequal(size(Yg), [2500, 3]), 'treeweave:test', 'unexpected output shape');
assert(o2.memory_usage() > 0, 'treeweave:test', 'memory_usage should be > 0');

% Transposed (struct-of-arrays) layout equals the AoS result transposed.
Yt = o2(Xg, 'transposed', true);
assert(isequal(size(Yt), [3, 2500]), 'treeweave:test', 'transposed shape wrong');
assert(max(max(abs(Yt' - Yg))) < 1e-12, 'treeweave:test', 'transposed != AoS transposed');
delete(o2);

fprintf('SUCCESS\n');

end
