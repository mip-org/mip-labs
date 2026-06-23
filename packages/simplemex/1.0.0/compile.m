function compile()
% Compile the simplemex MEX. cwd is the package source root.
%
% A deliberately trivial, single-threaded C MEX. The framework's
% setup_mex_compilers has already pointed `mex` at the right per-platform
% toolchain (MinGW + the static-linking mingw64.xml on Windows), so a plain
% mex() call suffices. The output is staged into matlab/, the only path entry.

fprintf('=== Compiling simplemex MEX ===\n');

src = fullfile('matlab', 'simplemex.c');
if ~exist(fullfile(pwd, src), 'file')
    error('simplemex:compile', '%s not found in %s', src, pwd);
end

mex(src, '-outdir', 'matlab');   % -> matlab/simplemex.<mexext>

out = fullfile('matlab', ['simplemex.' mexext]);
if ~exist(fullfile(pwd, out), 'file')
    error('simplemex:compile', 'expected %s after build, not found', out);
end

fprintf('=== simplemex MEX compiled: %s ===\n', out);

end
