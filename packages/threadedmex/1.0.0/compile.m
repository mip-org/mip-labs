function compile()
% Compile the threadedmex MEX (Windows / MinGW-w64, OpenMP). cwd is the package
% source root.
%
% Pattern mirrors fmmlib2d: compile the OpenMP worker directly with `gcc
% -fopenmp` (mex does not reliably apply -fopenmp to its own compile step), then
% mex-link the gateway against the worker object with -lgomp. setup_mex_compilers
% has already put MinGW-w64 first on PATH and selected mingw64.xml, which links
% -static, so libgomp is baked into the .mexw64 (no runtime DLL dependency).

fprintf('=== Compiling threadedmex MEX (Windows/MinGW-w64, OpenMP) ===\n');

matlabDir = fullfile(pwd, 'matlab');
worker    = fullfile(matlabDir, 'tmex_worker.c');
gateway   = fullfile(matlabDir, 'threadedmex.c');
for f = {worker, gateway}
    if ~exist(f{1}, 'file')
        error('threadedmex:compile', 'missing source %s', f{1});
    end
end

buildDir = fullfile(pwd, 'build_mex');
if ~exist(buildDir, 'dir'); mkdir(buildDir); end
obj = fullfile(buildDir, 'tmex_worker.o');

% Compile the OpenMP worker with gcc directly.
cmd = sprintf('gcc -O2 -fopenmp -c "%s" -o "%s"', worker, obj);
fprintf('%s\n', cmd);
[st, out] = system(cmd);
fprintf('%s', out);
if st ~= 0
    error('threadedmex:gcc', 'gcc -fopenmp failed (exit %d)', st);
end

% Directory holding libgomp.a (for -L), located via the active gcc.
[s, gdir] = system('gcc -print-file-name=libgomp.a');
gdir = strtrim(gdir);
if s == 0 && ~isempty(gdir)
    gdir = fileparts(gdir);
else
    gdir = '';
end

args = {'-largeArrayDims', gateway, obj};
if ~isempty(gdir)
    args{end+1} = ['-L' gdir];
end
args{end+1} = '-lgomp';
args{end+1} = '-lpthread';   % winpthreads; libgomp (posix MinGW) depends on it
args{end+1} = '-output';
args{end+1} = fullfile('matlab', 'threadedmex');
mex(args{:});

outFile = fullfile('matlab', ['threadedmex.' mexext]);
if ~exist(fullfile(pwd, outFile), 'file')
    error('threadedmex:compile', 'expected %s after build, not found', outFile);
end
fprintf('=== threadedmex compiled: %s ===\n', outFile);

end
