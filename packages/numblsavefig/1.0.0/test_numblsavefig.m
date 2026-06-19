% Test script for numblsavefig (run by `mip test numblsavefig`).
%
% Self-contained: it writes figures with numblsavefig and reads them back with
% base-MATLAB HDF5 functions (h5info/h5read/h5readatt), asserting the "numbl
% figure HDF5 layout v1" invariants that numbl's reader depends on — datasets
% vs. scalar attributes, grid orientation, NaN preservation, and patch face
% indexing. (Validation against numbl's TypeScript reader is done separately,
% outside MATLAB.)

fprintf('Testing numblsavefig...\n');

outdir = tempname;
mkdir(outdir);
cleanup = onCleanup(@() rmtree_quiet(outdir)); %#ok<NASGU>

% ── 1. plot: root + axes metadata, plot trace, NaN preservation ─────────────
fn = fullfile(outdir, 'plot.h5');
f = figure('Visible', 'off');
y = [1 4 NaN 16];                       % NaN must round-trip natively
plot(1:4, y, 'r--o', 'LineWidth', 2, 'MarkerSize', 8);
title('My Plot'); xlabel('xx'); ylabel('yy'); grid on;
numblsavefig(f, fn);
close(f);

assert(exist(fn, 'file') == 2, 'plot.h5 was not written');
assert(h5readatt(fn, '/', 'numbl_figure_version') == 1, 'bad version');
assert(strcmp(h5readatt(fn, '/', 'generator'), 'numblsavefig'), 'bad generator');
assert(h5readatt(fn, '/', 'current_axes') == 1, 'bad current_axes');
assert(strcmp(h5readatt(fn, '/axes/1', 'title'), 'My Plot'), 'bad title');
assert(strcmp(h5readatt(fn, '/axes/1', 'xlabel'), 'xx'), 'bad xlabel');
assert(h5readatt(fn, '/axes/1', 'grid_on') == 1, 'grid_on should be 1');

assert(strcmp(h5readatt(fn, '/axes/1/traces/0', 'kind'), 'plot'), 'bad kind');
xread = h5read(fn, '/axes/1/traces/0/x');
yread = h5read(fn, '/axes/1/traces/0/y');
assert(isequal(xread(:)', 1:4), 'x data mismatch');
assert(isequaln(yread(:)', y), 'y data (incl NaN) mismatch');

% scalar styling attributes must be true scalars, not length-1 arrays
lw = h5readatt(fn, '/axes/1/traces/0', 'lineWidth');
ms = h5readatt(fn, '/axes/1/traces/0', 'markerSize');
assert(isscalar(lw) && lw == 2, 'lineWidth must be scalar 2');
assert(isscalar(ms) && ms == 8, 'markerSize must be scalar 8');
col = h5readatt(fn, '/axes/1/traces/0', 'color');
assert(isequal(col(:)', [1 0 0]), 'color must be [1 0 0]');

% ── 2. surf: grid orientation (the #1 thing to get right) ───────────────────
fn = fullfile(outdir, 'surf.h5');
Z = [1 2 3; 4 5 6];                     % 2x3, distinct values
f = figure('Visible', 'off'); surf(Z); numblsavefig(f, fn); close(f);

assert(strcmp(h5readatt(fn, '/axes/1/traces/0', 'kind'), 'surf'), 'bad surf kind');
assert(h5readatt(fn, '/axes/1/traces/0', 'rows') == 2, 'surf rows');
assert(h5readatt(fn, '/axes/1/traces/0', 'cols') == 3, 'surf cols');
% numblsavefig writes the transpose so numbl reads a row-major [rows,cols] grid;
% base-MATLAB h5read returns that transpose, so transposing recovers Z exactly.
zread = h5read(fn, '/axes/1/traces/0/z');
assert(isequal(zread', Z), 'surf z orientation is wrong');

% ── 3. imagesc: 2-D CData grid + limit vectors ──────────────────────────────
fn = fullfile(outdir, 'imagesc.h5');
M = magic(4);
f = figure('Visible', 'off'); imagesc(M); numblsavefig(f, fn); close(f);
assert(strcmp(h5readatt(fn, '/axes/1/traces/0', 'kind'), 'imagesc'), 'bad imagesc kind');
zread = h5read(fn, '/axes/1/traces/0/z');
assert(isequal(zread', M), 'imagesc z orientation is wrong');

% ── 4. patch: ragged faces, 1-based->0-based, NaN->-1 ───────────────────────
fn = fullfile(outdir, 'patch.h5');
f = figure('Visible', 'off');
patch('Faces', [1 2 3 NaN; 1 3 4 5], ...
      'Vertices', [0 0; 1 0; 1 1; 0 1; 2 2], 'FaceColor', [0 1 0]);
numblsavefig(f, fn); close(f);
assert(strcmp(h5readatt(fn, '/axes/1/traces/0', 'kind'), 'patch'), 'bad patch kind');
fread_ = double(h5read(fn, '/axes/1/traces/0/faces'))';   % [N, M], numbl orientation
expected = [0 1 2 -1; 0 2 3 4];          % (Faces - 1), NaN -> -1
assert(isequal(fread_, expected), 'patch faces indexing is wrong');
isd = h5readatt(fn, '/axes/1/traces/0', 'is3D');
assert(isscalar(isd) && isd == 0, 'is3D must be scalar 0 for a 2-D patch');

% ── 5. defaults to gcf and overwrites an existing file ──────────────────────
fn = fullfile(outdir, 'gcf.h5');
fid = fopen(fn, 'w'); fwrite(fid, 'stale'); fclose(fid);   % pre-existing junk
f = figure('Visible', 'off'); bar([2 4 6]);
numblsavefig(fn);                        % no handle -> gcf; must overwrite
close(f);
assert(strcmp(h5readatt(fn, '/axes/1/traces/0', 'kind'), 'bar'), 'gcf/overwrite failed');

fprintf('SUCCESS\n');

function rmtree_quiet(d)
    if exist(d, 'dir')
        rmdir(d, 's');
    end
end
