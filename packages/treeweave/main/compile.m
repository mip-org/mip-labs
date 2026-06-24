function compile()
% Compile the treeweave MATLAB MEX (linux_x86_64, macos_arm64, windows_x86_64).
%
% compile.m runs with cwd set to the package source root (the fetched treeweave
% repo + overlaid channel files). It drives treeweave's OWN CMake build, which:
%   1. fetches the header-only deps (polyfit, POET, xsimd) via FetchContent,
%   2. compiles the checked-in mwrap gateway (bindings/matlab/generated/
%      treeweave_mex_gen.cpp + tw_*.m) — no mwrap, no bison/flex,
%   3. builds the C ABI static archive (treeweave_c_static), and
%   4. links the MEX via matlab_add_mex (statically pulling in treeweave_c_static
%      and, on GNU, -static-libstdc++/-static-libgcc).
%
% We then stage the generated tw_*.m + the compiled treeweave_mex.<ext> into
% bindings/matlab next to treeweave.m, so the package's single `paths:` entry
% (bindings/matlab) puts the whole binding on the MATLAB path.
%
% Build deps (cmake, a C++20 compiler) are provisioned by mip.yaml's per-OS
% `setup:`; see the Linux toolchain handling below.

fprintf('=== Compiling treeweave MATLAB MEX ===\n');

srcRoot = pwd;
if ~exist(fullfile(srcRoot, 'CMakeLists.txt'), 'file')
    error('treeweave:compile', 'CMakeLists.txt not found at %s', srcRoot);
end
if ~exist(fullfile(srcRoot, 'bindings', 'matlab', 'treeweave.mw'), 'file')
    error('treeweave:compile', 'bindings/matlab/treeweave.mw not found at %s', srcRoot);
end
bindingsDir = fullfile(srcRoot, 'bindings', 'matlab');

% MATLAB injects its own libcurl/libstdc++ onto LD_LIBRARY_PATH, which breaks the
% libcurl that CMake's FetchContent/CPM downloads shell out through (and the
% system git). Clear it for the duration of this script; onCleanup restores it.
if isunix && ~ismac
    origLdPath = getenv('LD_LIBRARY_PATH');
    setenv('LD_LIBRARY_PATH', '');
    restoreLdPath = onCleanup(@() setenv('LD_LIBRARY_PATH', origLdPath)); %#ok<NASGU>
end

% ---- Per-platform CMake configuration --------------------------------------
% TREEWEAVE_ARCH is a portable per-platform baseline; TREEWEAVE_C_MULTIARCH=ON
% makes the C ABI build every ISA variant of the family and pick one at runtime
% (no-op fan-out on MSVC / Apple-arm64, where it stays single-arch at the
% baseline). So the shipped binary loads on any CPU of the family yet still uses
% AVX2/AVX-512 (or NEON) where present.
defs = { ...
    '-DTREEWEAVE_BUILD_MATLAB=ON', ...
    '-DTREEWEAVE_BUILD_TESTS=OFF', ...
    '-DTREEWEAVE_BUILD_EXAMPLES=OFF', ...
    '-DTREEWEAVE_BUILD_C_API=ON', ...
    '-DTREEWEAVE_C_MULTIARCH=ON', ...
    '-DTREEWEAVE_WARNINGS_AS_ERRORS=OFF', ...
    '-DCMAKE_BUILD_TYPE=Release', ...
    sprintf('-DMatlab_ROOT_DIR=%s', matlabroot) };

genArg = '';   % CMake generator (Windows uses the VS multi-config generator)
if ismac
    defs{end+1} = '-DTREEWEAVE_ARCH=apple-m1';        % oldest Apple-Silicon baseline
    defs{end+1} = '-DCMAKE_OSX_ARCHITECTURES=arm64';
elseif ispc
    defs{end+1} = '-DTREEWEAVE_ARCH=x86-64';          % MSVC /arch: baseline (SSE2); multiarch fans out above it
    genArg = ' -G "Visual Studio 17 2022" -A x64';
    % No bison/flex/mwrap: the mwrap gateway is checked in upstream, so MSVC just
    % compiles it. TREEWEAVE_C_MULTIARCH=ON (above) gives the SSE2/AVX/AVX2/AVX512
    % runtime dispatch ladder via /arch: flags.
else
    defs{end+1} = '-DTREEWEAVE_ARCH=x86-64';          % portable x86-64 baseline + runtime dispatch
    % The build container's default gcc-toolset-10 predates C++20; switch to a
    % gcc-toolset >= 11 (installed via setup:). compile.m static-links its
    % libstdc++ so the MEX carries no newer-GLIBCXX dependency.
    [cc, cxx] = select_linux_compiler();
    defs{end+1} = sprintf('-DCMAKE_C_COMPILER=%s', cc);
    defs{end+1} = sprintf('-DCMAKE_CXX_COMPILER=%s', cxx);
end

buildDir = tempname;
mkdir(buildDir);
cleanupBuild = onCleanup(@() rmdir_silent(buildDir)); %#ok<NASGU>

% feature('numcores') re-probes the hardware; maxNumCompThreads is MATLAB's
% (CI-pinned to 1) thread cap.
nproc = feature('numcores');

cfgCmd = sprintf('cmake -S "%s" -B "%s"%s %s', srcRoot, buildDir, genArg, strjoin(defs, ' '));
run_or_error(cfgCmd, 'cmake configure');

buildCmd = sprintf('cmake --build "%s" --config Release --target treeweave_mex_matlab -j%d', ...
    buildDir, nproc);
run_or_error(buildCmd, 'cmake build');

% ---- Stage the generated stubs + the MEX into bindings/matlab --------------
ext = mexext;
mexFile = find_one(buildDir, ['treeweave_mex.' ext]);
if isempty(mexFile)
    error('treeweave:compile', 'treeweave_mex.%s not found under %s after build', ext, buildDir);
end
stubs = find_all(buildDir, 'tw_*.m');
if isempty(stubs)
    error('treeweave:compile', 'mwrap-generated tw_*.m stubs not found under %s', buildDir);
end

copyfile(mexFile, fullfile(bindingsDir, ['treeweave_mex.' ext]));
for k = 1:numel(stubs)
    [~, nm, xt] = fileparts(stubs{k});
    copyfile(stubs{k}, fullfile(bindingsDir, [nm xt]));
end
fprintf('Staged treeweave_mex.%s + %d tw_*.m stubs into %s\n', ext, numel(stubs), bindingsDir);

% ---- Linux: rewrite absolute MATLAB DT_NEEDED entries to basenames ---------
% matlab_add_mex links the .mexa64 against MATLAB's libmex/libmx, which ship
% without a DT_SONAME, so the link can bake the CI runner's absolute MATLAB paths
% into DT_NEEDED. Those paths don't exist on an end-user machine. Rewrite each
% absolute NEEDED entry to its basename (resolved via the user's MATLAB at load
% time); drop libMatlabEngine (classic mx/mex-API code never calls it).
if isunix && ~ismac
    patch_mex_needed(fullfile(bindingsDir, ['treeweave_mex.' ext]));
end

fprintf('=== treeweave MATLAB MEX compilation complete ===\n');

end


function [cc, cxx] = select_linux_compiler()
% Return a C/C++ compiler pair that supports C++20 (>= gcc 11). Prefer the newest
% gcc-toolset under /opt/rh; otherwise fall back to the default cc/c++ if it is
% already new enough (e.g. a non-container build host).
best = '';
bestVer = -1;
d = dir('/opt/rh/gcc-toolset-*');
for i = 1:numel(d)
    tok = regexp(d(i).name, 'gcc-toolset-(\d+)$', 'tokens', 'once');
    if isempty(tok); continue; end
    ver = str2double(tok{1});
    gxx = fullfile('/opt/rh', d(i).name, 'root', 'usr', 'bin', 'g++');
    if ver >= 11 && ver > bestVer && exist(gxx, 'file')
        bestVer = ver;
        best = fullfile('/opt/rh', d(i).name, 'root', 'usr', 'bin');
    end
end

if ~isempty(best)
    setenv('PATH', [best ':' getenv('PATH')]);
    cc  = fullfile(best, 'gcc');
    cxx = fullfile(best, 'g++');
    fprintf('Using gcc-toolset >= 11 at %s\n', best);
    return;
end

% No gcc-toolset: accept the default g++ only if it is >= 11.
[st, out] = system('g++ -dumpfullversion -dumpversion');
major = NaN;
if st == 0
    tok = regexp(strtrim(out), '^(\d+)', 'tokens', 'once');
    if ~isempty(tok); major = str2double(tok{1}); end
end
if ~isnan(major) && major >= 11
    cc  = 'gcc';
    cxx = 'g++';
    fprintf('Using default gcc %d (>= 11)\n', major);
    return;
end

error('treeweave:compile', ...
    ['No C++20-capable GCC found: need gcc-toolset >= 11 under /opt/rh or a ' ...
     'default g++ >= 11 (found "%s"). Install gcc-toolset via mip.yaml setup:.'], strtrim(out));
end


function patch_mex_needed(mexPath)
% Rewrite absolute-path DT_NEEDED entries to basenames; drop libMatlabEngine.
[st, needed] = system(sprintf('patchelf --print-needed "%s"', mexPath));
if st ~= 0
    fprintf('  patchelf --print-needed failed (exit %d); skipping NEEDED rewrite\n', st);
    return;
end
lines = splitlines(strtrim(needed));
for i = 1:numel(lines)
    entry = strtrim(lines{i});
    if isempty(entry); continue; end
    if startsWith(entry, '/')
        [~, nm, xt] = fileparts(entry);
        base = [nm xt];
        run_or_error(sprintf('patchelf --replace-needed "%s" "%s" "%s"', entry, base, mexPath), ...
            sprintf('patchelf rewrite %s', base));
        entry = base;
    end
    if strcmp(entry, 'libMatlabEngine.so')
        run_or_error(sprintf('patchelf --remove-needed "%s" "%s"', entry, mexPath), ...
            'patchelf remove libMatlabEngine.so');
    end
end
end


function p = find_one(root, pattern)
hits = find_all(root, pattern);
if isempty(hits); p = ''; else; p = hits{1}; end
end


function hits = find_all(root, pattern)
% Recursive file search for `pattern` under `root` (cross-platform; avoids shell
% globbing differences between find and dir).
hits = {};
stack = {root};
while ~isempty(stack)
    cur = stack{end};
    stack(end) = [];
    listing = dir(cur);
    for i = 1:numel(listing)
        if strcmp(listing(i).name, '.') || strcmp(listing(i).name, '..')
            continue;
        end
        full = fullfile(cur, listing(i).name);
        if listing(i).isdir
            stack{end+1} = full; %#ok<AGROW>
        elseif ~isempty(regexp(listing(i).name, ['^' regexptranslate('wildcard', pattern) '$'], 'once'))
            hits{end+1} = full; %#ok<AGROW>
        end
    end
end
end


function run_or_error(cmd, what)
fprintf('  [%s]\n    %s\n', what, cmd);
[st, out] = system(cmd);
fprintf('%s', out);
if st ~= 0
    error('treeweave:compile', '%s failed (exit %d)', what, st);
end
end


function rmdir_silent(d)
if exist(d, 'dir')
    try
        rmdir(d, 's');
    catch
    end
end
end
