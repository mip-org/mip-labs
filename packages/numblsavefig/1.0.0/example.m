% Minimal end-to-end example for numblsavefig.
%
%   mip install --channel mip-org/labs numblsavefig
%   mip load numblsavefig

% Build a figure as usual.
figure;
surf(peaks(40));
shading interp; colormap(parula); colorbar;
title('peaks surface'); xlabel('X'); ylabel('Y'); zlabel('Z');
view(-37.5, 30);

% Save it to numbl's HDF5 figure format.
numblsavefig(gcf, 'peaks.h5');

% `peaks.h5` can now be opened in the numbl figure viewer
% (https://concept-collection.github.io/numbl-figure-viewer/ — drag it in),
% imported via importFigureHdf5, or opened in the numbl IDE.
fprintf('Wrote peaks.h5\n');
