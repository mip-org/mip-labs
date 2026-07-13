function test_mlxshake()
% Channel test for mlxshake. Runs after `mip load mlxshake`.
%
% Two parts:
%   1. Display-independent checks (always run, incl. headless CI): the public
%      API resolves, and constructing MlxshakeBase succeeds. That construction
%      is what triggers the library initializer -> Log4jConfigurator, so it is
%      the regression guard for the channel's log4j patch (see README.md);
%      without the patch it crashes on recent MATLAB.
%   2. Live Editor export (display-dependent): exports the bundled
%      test_fixture.mlx to LaTeX and HTML. matlab.internal.liveeditor requires
%      a display, and the channel's `any` runner is headless, so these checks
%      are skipped when no DISPLAY is available. They run wherever one is (local
%      desktop / a runner with an X server).
%
% Markdown export is intentionally never tested: on recent MATLAB it fails
% because upstream's LaTeX->Markdown image handling expects a "<stem>_images"
% figure folder while modern MATLAB emits "<stem>_media" (see README.md).

fprintf('Testing mlxshake...\n');

% --- 1. Display-independent checks ---

% Public API resolves (namespaced members: use which, since
% exist('pkg.fcn','file') can report 0 for +package members).
for name = ["janklab.mlxshake.exportlivescript", ...
            "janklab.mlxshake.mlx2latex", ...
            "janklab.mlxshake.lslatex2markdown", ...
            "janklab.mlxshake.MlxExportOptions"]
    assert(~isempty(which(char(name))), 'API not found on path: %s', name);
end

% Constructing MlxshakeBase runs the library initializer, exercising the
% patched Log4jConfigurator. Errors here on recent MATLAB without the patch.
janklab.mlxshake.internal.MlxshakeBase();
fprintf('Library initializer OK (log4j patch effective).\n');

% --- 2. Live Editor export (needs a display) ---

if isunix && ~ismac && isempty(getenv('DISPLAY'))
    fprintf(['No DISPLAY (headless runner): skipping Live Editor export ' ...
             'checks.\nSUCCESS\n']);
    return
end

thisDir = fileparts(mfilename('fullpath'));
mlx = fullfile(thisDir, 'test_fixture.mlx');
assert(exist(mlx, 'file') == 2, 'test_fixture.mlx not found next to the test');

outDir = tempname;
mkdir(outDir);
cleanup = onCleanup(@() rmdir(outDir, 's')); %#ok<NASGU>

% mlx2latex: .mlx -> .tex
texFile = fullfile(outDir, 'fixture.tex');
janklab.mlxshake.mlx2latex(mlx, texFile);
assertNonEmptyFile(texFile, 'mlx2latex .tex output');

% exportlivescript to the direct-export formats.
exportTo(mlx, fullfile(outDir, 'fixture_ls.tex'), 'latex');
exportTo(mlx, fullfile(outDir, 'fixture.html'), 'html');

fprintf('SUCCESS\n');

end

function exportTo(mlx, outFile, format)
    opts = janklab.mlxshake.MlxExportOptions;
    opts.format = format;
    opts.outFile = outFile;
    janklab.mlxshake.exportlivescript(mlx, opts);
    assertNonEmptyFile(outFile, sprintf('exportlivescript %s output', format));
end

function assertNonEmptyFile(path, what)
    d = dir(path);
    assert(~isempty(d) && d(1).bytes > 0, ...
        '%s missing or empty: %s', what, path);
end
