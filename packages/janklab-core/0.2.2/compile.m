function compile()
% Compile janklab-core's single MEX. cwd is the package source root.
%
% Only one native source ships: Mcode/classes/+jl/+algo/binsearch_mex.c, a
% self-contained mex.h binary search with no third-party dependencies. The
% framework's setup_mex_compilers has already pointed `mex` at the right
% per-platform toolchain (MinGW + the static-linking mingw64.xml on Windows),
% so a plain mex() call suffices. The output is staged next to the source so
% jl.algo.binsearch resolves jl.algo.binsearch_mex through the +jl/+algo
% namespace.

fprintf('=== Compiling janklab-core binsearch MEX ===\n');

outdir = fullfile('Mcode', 'classes', '+jl', '+algo');
src    = fullfile(outdir, 'binsearch_mex.c');
if ~exist(fullfile(pwd, src), 'file')
    error('janklab:compile', '%s not found in %s', src, pwd);
end

if isunix && ~ismac
    % Statically link libgcc so the .mexa64 depends only on OS-provided
    % libraries across the RHEL/Rocky fleet.
    mex('LDFLAGS=$LDFLAGS -static-libgcc', src, '-outdir', outdir);
else
    mex(src, '-outdir', outdir);
end

out = fullfile(outdir, ['binsearch_mex.' mexext]);
if ~exist(fullfile(pwd, out), 'file')
    error('janklab:compile', 'expected %s after build, not found', out);
end

fprintf('=== binsearch MEX compiled: %s ===\n', out);

end
