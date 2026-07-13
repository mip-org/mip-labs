function test_janklab_core_mex()
% Channel test for janklab-core on the native (MEX) builds. Runs after
% `mip load janklab-core`.
%
% Exercises the shipped binsearch MEX (satisfying the MEX-coverage gate) via
% jl.algo.binsearch on both the double and single code paths, plus a sampling
% of the pure-MATLAB utility layer. Intentionally does NOT call init_janklab:
% mip only sets up the MATLAB path, and janklab's Java-backed features are out
% of scope for this build's smoke test (see README.md).

fprintf('Testing janklab-core (MEX build)...\n');

% --- binsearch MEX: double path ---
x = [1 3 5 7 9 11];
[ix, ix2] = jl.algo.binsearch(7, x);
assert(isequal(ix, 4) && isequal(ix2, 4), ...
    'binsearch(7, [1 3 5 7 9 11]) should find index 4');
[ix, ix2] = jl.algo.binsearch(8, x);
assert(isempty(ix) && isequal(ix2, 5), ...
    'binsearch(8, ...) should miss and report insertion index 5');

% --- binsearch MEX: single path ---
[ixs, ~] = jl.algo.binsearch(single(5), single(x));
assert(isequal(ixs, 3), 'binsearch(single(5), ...) should find index 3');

% Confirm the MEX (not the mcode fallback) actually loaded. This mirrors the
% coverage gate's own check, but asserting it here gives a clearer failure if
% the double/single dispatch ever stops routing through the MEX.
[~, loadedPaths] = inmem('-completenames');
names = cell(size(loadedPaths));
for i = 1:numel(loadedPaths)
    [~, names{i}] = fileparts(loadedPaths{i});
end
assert(any(strcmp(names, 'binsearch_mex')), ...
    'binsearch_mex should be loaded in memory after the double/single calls');

% --- pure-MATLAB utility layer ---
assert(isequal(qw('foo bar baz'), {'foo', 'bar', 'baz'}), 'qw split failed');
assert(isequal(ifthen(true, 1, 2), 1), 'ifthen(true,...) failed');
assert(isequal(ifthen(false, 1, 2), 2), 'ifthen(false,...) failed');
assert(isequal(firstNonEmpty([], [], 3), 3), 'firstNonEmpty failed');
mustBeStringy("ok");   % throws on failure

fprintf('SUCCESS\n');

end
