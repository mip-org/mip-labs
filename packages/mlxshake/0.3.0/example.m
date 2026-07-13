% Minimal mlxshake usage example: export a MATLAB Live Script to HTML.
%
% See README.md for the current status of the Markdown format on recent MATLAB.

mip install --channel mip-org/labs mlxshake
mip load mlxshake

% Export a Live Script (.mlx) directly to HTML, next to the input file.
opts = janklab.mlxshake.MlxExportOptions;
opts.format = 'html';
opts.outFile = 'MyLiveScript.html';
outFile = janklab.mlxshake.exportlivescript('MyLiveScript.mlx', opts);
fprintf('Wrote %s\n', outFile);

% Or just produce the intermediate LaTeX (.tex + matlab.sty + images):
janklab.mlxshake.mlx2latex('MyLiveScript.mlx', 'MyLiveScript.tex');
