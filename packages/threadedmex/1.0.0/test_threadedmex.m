function test_threadedmex()
% Channel test for threadedmex. Runs after `mip load threadedmex`.
%
% Invokes the shipped MEX (satisfying the MEX-coverage gate) and checks
% correctness. Also reports the OpenMP thread count (not asserted > 1, since the
% CI runner's core count / OMP policy may vary).

fprintf('Testing threadedmex...\n');

% Scalar.
assert(threadedmex(0) == 1, 'threadedmex(0) should be 1');

% Array preserves shape and adds 1 element-wise.
x = reshape(1:12, 3, 4);
[y, nthreads] = threadedmex(x);
assert(isequal(size(y), size(x)), 'threadedmex must preserve input shape');
assert(isequal(y, x + 1), 'threadedmex must add 1 element-wise');
fprintf('threadedmex used %d OpenMP thread(s)\n', nthreads);
assert(nthreads >= 1, 'thread count should be at least 1');

% Error path: non-double input is rejected (still exercises the MEX).
errid = '';
try
    threadedmex(int32(3));
catch ME
    errid = ME.identifier;
end
assert(strcmp(errid, 'threadedmex:type'), ...
    'threadedmex should reject non-double input with threadedmex:type');

fprintf('SUCCESS\n');

end
