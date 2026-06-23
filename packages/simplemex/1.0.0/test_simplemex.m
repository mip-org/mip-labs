function test_simplemex()
% Channel test for simplemex. Runs after `mip load simplemex`.
%
% Invokes the shipped MEX (satisfying the MEX-coverage gate) and checks
% correctness, including shape preservation and the input-type error path.

fprintf('Testing simplemex...\n');

% Scalars.
assert(simplemex(0) == 1, 'simplemex(0) should be 1');
assert(simplemex(41) == 42, 'simplemex(41) should be 42');
assert(abs(simplemex(-1.5) - (-0.5)) < 1e-12, 'simplemex(-1.5) should be -0.5');

% Arrays preserve shape and add 1 element-wise.
x = [1 2 3; 4 5 6];
y = simplemex(x);
assert(isequal(size(y), size(x)), 'simplemex must preserve input shape');
assert(isequal(y, x + 1), 'simplemex must add 1 element-wise');

% Error path: non-double input is rejected (still counts as exercising the MEX).
errid = '';
try
    simplemex(int32(3));
catch ME
    errid = ME.identifier;
end
assert(strcmp(errid, 'simplemex:type'), ...
    'simplemex should reject non-double input with simplemex:type');

fprintf('SUCCESS\n');

end
