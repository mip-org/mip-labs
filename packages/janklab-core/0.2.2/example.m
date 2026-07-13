% Minimal janklab-core usage example.
%
% Covers the pure-MATLAB layer that is available immediately after `mip load`
% (no init_janklab needed). For the Java-backed features (jl.sql / MDBC, Excel
% I/O, FTP, logging), run init_janklab after loading — see README.md.

mip install --channel mip-org/labs janklab-core
mip load janklab-core

% qw: split a string on whitespace into a cellstr (Perl-style "quote words").
words = qw('foo bar baz');
disp(words);                       % {'foo','bar','baz'}

% ifthen: inline conditional value selection.
label = ifthen(numel(words) > 2, 'many', 'few');
fprintf('label = %s\n', label);    % many

% binsearch: binary search over a sorted numeric vector (uses the MEX on the
% native builds, the mcode fallback elsewhere).
x = [1 3 5 7 9 11];
[ix, insertAt] = jl.algo.binsearch(7, x);
fprintf('found 7 at index %d\n', ix);          % 4
[ix, insertAt] = jl.algo.binsearch(8, x);
fprintf('8 not found; would insert at %d\n', insertAt);   % 5
