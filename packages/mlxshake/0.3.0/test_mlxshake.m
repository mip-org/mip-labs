function test_mlxshake()
% Channel test for mlxshake. Runs after `mip load mlxshake`.
%
% Verifies the public API is on the path, then exports the bundled
% test_fixture.mlx to LaTeX and HTML (the "direct" export formats) and via
% mlx2latex, asserting each produces a non-empty output file. Constructing
% MlxExportOptions also exercises the library initializer, which is where the
% channel's log4j patch (see README.md) takes effect.
%
% Markdown export is intentionally NOT tested: on recent MATLAB it fails
% because upstream's LaTeX->Markdown image handling expects a "<stem>_images"
% figure folder while modern MATLAB emits "<stem>_media" (see README.md).

fprintf('Testing mlxshake...\n');

% Public API is resolvable (namespaced functions/classes: use which, since
% exist('pkg.fcn','file') can report 0 for +package members).
for name = ["janklab.mlxshake.exportlivescript", ...
            "janklab.mlxshake.mlx2latex", ...
            "janklab.mlxshake.lslatex2markdown", ...
            "janklab.mlxshake.MlxExportOptions"]
    assert(~isempty(which(char(name))), 'API not found on path: %s', name);
end

% Locate the bundled fixture relative to this test file (robust to cwd).
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
