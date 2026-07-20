% treeweave — minimal end-to-end example.
%
% treeweave builds an adaptive tree of polynomial leaves that approximates a
% function over a box domain to a requested tolerance, then evaluates it fast.

mip install --channel mip-org/labs treeweave
mip load treeweave

% 1-D scalar fit on [0, 1]. dim & out_dim are inferred.
f   = @(x) exp(0.5*x(1)) + sin(3*x(1));
obj = treeweave(f, 0, 1, 1e-8);

X = linspace(0, 1, 1000)';
Y = obj.eval(X);                      % N x out_dim   (or: Y = obj(X))

fprintf('1-D max error: %.2e\n', max(abs(Y - (exp(0.5*X) + sin(3*X)))));
delete(obj);

% 2-D -> 3-D vector fit. out_dim is inferred by probing g at the box midpoint.
g    = @(x) [sin(x(1)+x(2)); cos(x(1)-x(2)); x(1)*x(2)];
obj2 = treeweave(g, [-1, -1], [1, 1], 1e-6, 'max_memory_mib', 64);

[gx, gy] = meshgrid(linspace(-1, 1, 50));
Y2 = obj2([gx(:), gy(:)]);            % 2500 x 3
fprintf('Memory: %.1f KiB\n', obj2.memory_usage() / 1024);
delete(obj2);
