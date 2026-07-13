function test_janklab_core()
% Channel test for janklab-core on the [any] fallback build (no MEX). Runs
% after `mip load janklab-core`.
%
% Exercises only the pure-MATLAB layer: the binsearch mcode fallback plus a
% sampling of the utility functions. Does NOT touch jl.algo.binsearch on
% double/single (that path needs the MEX, which this build does not ship) and
% does NOT call init_janklab (Java-backed features are out of scope here).

fprintf('Testing janklab-core (pure-MATLAB build)...\n');

% --- binsearch mcode fallback (no MEX) ---
x = [1 3 5 7 9 11];
[ix, ix2] = jl.algo.binsearch_mcode(7, x);
assert(isequal(ix, 4) && isequal(ix2, 4), ...
    'binsearch_mcode(7, ...) should find index 4');
[ix, ix2] = jl.algo.binsearch_mcode(8, x);
assert(isempty(ix) && isequal(ix2, 5), ...
    'binsearch_mcode(8, ...) should miss and report insertion index 5');

% --- pure-MATLAB utility layer ---
assert(isequal(qw('foo bar baz'), {'foo', 'bar', 'baz'}), 'qw split failed');
assert(isequal(ifthen(true, 1, 2), 1), 'ifthen(true,...) failed');
assert(isequal(ifthen(false, 1, 2), 2), 'ifthen(false,...) failed');
assert(isequal(firstNonEmpty([], [], 3), 3), 'firstNonEmpty failed');
mustBeStringy("ok");   % throws on failure

fprintf('SUCCESS\n');

end
