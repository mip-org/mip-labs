function numblsavefig(varargin)
%NUMBLSAVEFIG Save a MATLAB figure to numbl's HDF5 figure format.
%   NUMBLSAVEFIG(FIG, FILENAME) walks the graphics-object tree of figure FIG
%   and writes a "numbl figure HDF5 layout v1" file FILENAME (.h5) that numbl
%   can open in its figure viewer / importFigureHdf5 / IDE.
%
%   NUMBLSAVEFIG(FILENAME) saves the current figure (GCF).
%
%   This is the MATLAB->numbl complement to numbl's own HDF5 export. It reads
%   the live object tree via the public graphics API (get/properties), which is
%   far more robust than parsing the undocumented .fig serialization.
%
%   Examples:
%       plot(1:10, (1:10).^2); numblsavefig('plot.h5')
%       surf(peaks);            numblsavefig(gcf, 'surf.h5')
%
%   Supported (P0/P1): single or multiple axes; line/plot/plot3, surf/mesh,
%   imagesc, bar/barh, patch, scatter; axes title/labels/limits, log scales,
%   grid, box, colormap, caxis, YDir, 3-D view, legend. Unsupported objects are
%   skipped with a warning. NaN/Inf round-trip natively.
%
%   Tested with MATLAB R2025b (HG2). Base MATLAB only (no toolbox).
%
%   See also: GCF, H5CREATE, H5WRITE, H5WRITEATT.

    % ---- argument parsing --------------------------------------------------
    if nargin == 1
        fig = gcf;
        filename = varargin{1};
    elseif nargin == 2
        fig = varargin{1};
        filename = varargin{2};
    else
        error('numblsavefig:args', ...
            'Usage: numblsavefig(fig, filename) or numblsavefig(filename)');
    end
    filename = char(filename);
    if isempty(filename)
        error('numblsavefig:args', 'FILENAME must be non-empty.');
    end
    if ~(isa(fig, 'matlab.ui.Figure') || (isnumeric(fig) && isscalar(fig)))
        error('numblsavefig:args', 'FIG must be a figure handle.');
    end

    % ---- (re)create the output file fresh (overwrite) ----------------------
    if exist(filename, 'file')
        delete(filename);
    end
    fid = H5F.create(filename, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    H5F.close(fid);

    % ---- root attributes ---------------------------------------------------
    write_num(filename, '/', 'numbl_figure_version', 1);
    h5writeatt(filename, '/', 'generator', 'numblsavefig');
    write_num(filename, '/', 'current_axes', 1);
    sg = super_title(fig);
    if ~isempty(sg)
        h5writeatt(filename, '/', 'sgtitle', sg);
    end

    % ---- collect axes (in creation order) ----------------------------------
    axList = findobj(fig, 'Type', 'axes');
    axList = flipud(axList(:));   % findobj returns reverse creation order
    if isempty(axList)
        warning('numblsavefig:noaxes', 'Figure has no axes; nothing to save.');
        return;
    end

    for i = 1:numel(axList)
        write_axes(filename, sprintf('/axes/%d', i), axList(i));
    end
end

% ===========================================================================
%  Axes
% ===========================================================================
function write_axes(fn, axPath, ax)
    ensure_group(fn, axPath);

    is3d = axes_is_3d(ax);

    % --- text / labels ---
    write_str_attr(fn, axPath, 'title',  text_of(ax.Title));
    write_str_attr(fn, axPath, 'xlabel', text_of(ax.XLabel));
    write_str_attr(fn, axPath, 'ylabel', text_of(ax.YLabel));
    if is3d
        write_str_attr(fn, axPath, 'zlabel', text_of(ax.ZLabel));
    end

    % --- limits (always concrete in MATLAB) ---
    h5writeatt(fn, axPath, 'xlim', double(ax.XLim(:).'));
    h5writeatt(fn, axPath, 'ylim', double(ax.YLim(:).'));
    if is3d
        h5writeatt(fn, axPath, 'zlim', double(ax.ZLim(:).'));
    end

    % --- scales -> axis_scale ---
    h5writeatt(fn, axPath, 'axis_scale', scale_string(ax.XScale, ax.YScale));

    % --- grid / box / ydir ---
    write_num(fn, axPath, 'grid_on', ...
        strcmp(ax.XGrid, 'on') || strcmp(ax.YGrid, 'on'));
    write_num(fn, axPath, 'box_on', strcmp(ax.Box, 'on'));
    h5writeatt(fn, axPath, 'y_dir', char(ax.YDir));

    % --- colormap + caxis ---
    cmap = ax.Colormap;
    if ~isempty(cmap) && size(cmap, 2) == 3
        write_grid(fn, [axPath '/colormap_data'], double(cmap), 'double');
    end
    h5writeatt(fn, axPath, 'caxis', double(ax.CLim(:).'));

    % --- 3-D view ---
    if is3d
        v = ax.View;
        write_num(fn, axPath, 'view_az', v(1));
        write_num(fn, axPath, 'view_el', v(2));
    end

    % --- legend (string[]) ---
    lg = legend_for_axes(ax);
    if ~isempty(lg)
        h5writeatt(fn, axPath, 'legend', lg);
    end

    % --- traces, in draw order (Children is reverse draw order) ---
    children = flipud(ax.Children(:));
    k = 0;
    for c = 1:numel(children)
        if write_trace(fn, axPath, k, children(c))
            k = k + 1;
        end
    end
end

% ===========================================================================
%  Traces -- returns true if a trace group was written
% ===========================================================================
function wrote = write_trace(fn, axPath, k, obj)
    wrote = false;
    tp = sprintf('%s/traces/%d', axPath, k);
    switch obj.Type
        case 'line'
            wrote = trace_line(fn, tp, obj);
        case 'scatter'
            wrote = trace_scatter(fn, tp, obj);
        case 'surface'
            wrote = trace_surface(fn, tp, obj);
        case 'image'
            wrote = trace_image(fn, tp, obj);
        case 'bar'
            wrote = trace_bar(fn, tp, obj);
        case 'patch'
            wrote = trace_patch(fn, tp, obj);
        otherwise
            if ~ismember(obj.Type, {'text', 'legend', 'colorbar'})
                warning('numblsavefig:unsupported', ...
                    'Skipping unsupported object of Type ''%s''.', obj.Type);
            end
    end
end

function wrote = trace_line(fn, tp, obj)
    x = obj.XData; y = obj.YData; z = obj.ZData;
    if isempty(x) || isempty(y)
        wrote = false; return;
    end
    if ~isempty(z) && any(z(:) ~= z(1))
        ensure_group(fn, tp);
        h5writeatt(fn, tp, 'kind', 'plot3');
        write_1d(fn, [tp '/x'], x);
        write_1d(fn, [tp '/y'], y);
        write_1d(fn, [tp '/z'], z);
    else
        ensure_group(fn, tp);
        h5writeatt(fn, tp, 'kind', 'plot');
        write_1d(fn, [tp '/x'], x);
        write_1d(fn, [tp '/y'], y);
    end
    write_line_style(fn, tp, obj);
    wrote = true;
end

function wrote = trace_scatter(fn, tp, obj)
    x = obj.XData; y = obj.YData; z = obj.ZData;
    if isempty(x) || isempty(y)
        wrote = false; return;
    end
    ensure_group(fn, tp);
    if ~isempty(z)
        h5writeatt(fn, tp, 'kind', 'plot3');
        write_1d(fn, [tp '/x'], x);
        write_1d(fn, [tp '/y'], y);
        write_1d(fn, [tp '/z'], z);
    else
        h5writeatt(fn, tp, 'kind', 'plot');
        write_1d(fn, [tp '/x'], x);
        write_1d(fn, [tp '/y'], y);
    end
    h5writeatt(fn, tp, 'lineStyle', 'none');
    mk = obj.Marker; if strcmp(mk, 'none'); mk = 'o'; end
    h5writeatt(fn, tp, 'marker', mk);
    % per-point CData/SizeData collapse to a single color/size (documented loss)
    c = obj.CData;
    if isnumeric(c) && size(c, 2) == 3 && ~isempty(c)
        h5writeatt(fn, tp, 'color', double(c(1, :)));
    end
    sd = obj.SizeData;
    if isnumeric(sd) && ~isempty(sd)
        % scatter SizeData is points^2; markerSize is a diameter-like value
        write_num(fn, tp, 'markerSize', sqrt(double(sd(1))));
    end
    wrote = true;
end

function wrote = trace_surface(fn, tp, obj)
    Z = double(obj.ZData);
    if isempty(Z)
        wrote = false; return;
    end
    [rows, cols] = size(Z);
    [X, Y] = surface_grids(obj, rows, cols);

    ensure_group(fn, tp);
    % MATLAB `mesh` produces a surface with a solid (white) FaceColor and a
    % colormapped EdgeColor ('flat'/'interp'); `surf` is the reverse
    % (FaceColor 'flat', solid EdgeColor). Emit a wireframe (faceColor 'none')
    % for the mesh case so numbl renders it as a mesh, not a solid surface.
    faceColor = obj.FaceColor;
    edgeColor = obj.EdgeColor;
    isMesh = isnumeric(faceColor) && ischar(edgeColor) && ...
        any(strcmp(edgeColor, {'flat', 'interp'}));
    if isMesh
        h5writeatt(fn, tp, 'kind', 'mesh');
        faceColor = 'none';
    else
        h5writeatt(fn, tp, 'kind', 'surf');
    end
    write_grid(fn, [tp '/x'], X, 'double');
    write_grid(fn, [tp '/y'], Y, 'double');
    write_grid(fn, [tp '/z'], Z, 'double');
    C = obj.CData;
    if isnumeric(C) && isequal(size(C), [rows cols])
        write_grid(fn, [tp '/c'], double(C), 'double');
    end
    write_num(fn, tp, 'rows', rows);
    write_num(fn, tp, 'cols', cols);
    write_color_attr(fn, tp, 'faceColor', faceColor);
    write_color_attr(fn, tp, 'edgeColor', edgeColor);
    if isnumeric(obj.FaceAlpha)
        write_num(fn, tp, 'faceAlpha', obj.FaceAlpha);
    end
    wrote = true;
end

function wrote = trace_image(fn, tp, obj)
    C = obj.CData;
    if isempty(C) || ~ismatrix(C)
        % truecolor RGB images are not supported by the imagesc trace
        if ~ismatrix(C)
            warning('numblsavefig:rgbimage', ...
                'Skipping truecolor image (CData is M-by-N-by-3).');
        end
        wrote = false; return;
    end
    C = double(C);
    [rows, cols] = size(C);
    ensure_group(fn, tp);
    h5writeatt(fn, tp, 'kind', 'imagesc');
    write_grid(fn, [tp '/z'], C, 'double');
    write_1d(fn, [tp '/x'], [obj.XData(1) obj.XData(end)]);
    write_1d(fn, [tp '/y'], [obj.YData(1) obj.YData(end)]);
    write_num(fn, tp, 'rows', rows);
    write_num(fn, tp, 'cols', cols);
    wrote = true;
end

function wrote = trace_bar(fn, tp, obj)
    x = obj.XData; y = obj.YData;
    if isempty(x) || isempty(y)
        wrote = false; return;
    end
    ensure_group(fn, tp);
    if strcmp(obj.Horizontal, 'on')
        h5writeatt(fn, tp, 'kind', 'barh');
    else
        h5writeatt(fn, tp, 'kind', 'bar');
    end
    write_1d(fn, [tp '/x'], x);
    write_1d(fn, [tp '/y'], y);
    write_num(fn, tp, 'width', obj.BarWidth);
    write_color_attr(fn, tp, 'color', obj.FaceColor);
    wrote = true;
end

function wrote = trace_patch(fn, tp, obj)
    V = obj.Vertices; F = obj.Faces;
    if isempty(V) || isempty(F)
        wrote = false; return;
    end
    ensure_group(fn, tp);
    h5writeatt(fn, tp, 'kind', 'patch');
    write_grid(fn, [tp '/vertices'], double(V), 'double');
    % MATLAB faces are 1-based, padded with NaN -> numbl wants 0-based int -1 pad
    F0 = F - 1;
    F0(isnan(F0)) = -1;
    write_grid(fn, [tp '/faces'], int32(F0), 'int32');
    write_num(fn, tp, 'is3D', size(V, 2) >= 3);
    write_color_attr(fn, tp, 'faceColor', obj.FaceColor);
    write_color_attr(fn, tp, 'edgeColor', obj.EdgeColor);
    if isnumeric(obj.FaceAlpha)
        write_num(fn, tp, 'faceAlpha', obj.FaceAlpha);
    end
    wrote = true;
end

% ===========================================================================
%  Shared trace helpers
% ===========================================================================
function write_line_style(fn, tp, obj)
    write_color_attr(fn, tp, 'color', obj.Color);
    h5writeatt(fn, tp, 'lineStyle', char(obj.LineStyle));
    h5writeatt(fn, tp, 'marker', char(obj.Marker));
    write_num(fn, tp, 'lineWidth', obj.LineWidth);
    write_num(fn, tp, 'markerSize', obj.MarkerSize);
end

function write_color_attr(fn, path, name, val)
    % Numeric RGB -> [r g b] double; keyword ('flat'/'interp'/'none') -> string.
    if isnumeric(val) && numel(val) == 3
        h5writeatt(fn, path, name, double(val(:).'));
    elseif ischar(val) || isstring(val)
        h5writeatt(fn, path, name, char(val));
    end
end

function [X, Y] = surface_grids(obj, rows, cols)
    % MATLAB surf XData/YData may be vectors (length cols / rows) or full grids.
    X = obj.XData; Y = obj.YData;
    if isvector(X) || isvector(Y)
        xv = X(:).'; yv = Y(:).';
        if isempty(xv); xv = 1:cols; end
        if isempty(yv); yv = 1:rows; end
        [X, Y] = meshgrid(xv, yv);   % -> [rows, cols]
    end
    X = double(X); Y = double(Y);
end

% ===========================================================================
%  HDF5 write primitives
% ===========================================================================
function write_1d(fn, path, v)
    % 1-D dataset. A scalar h5create size yields a true rank-1 dataspace
    % (a [1 n] / [n 1] size would create an unwanted 2-D dataset).
    v = double(v(:).');
    if isempty(v); return; end
    h5create(fn, path, numel(v));
    h5write(fn, path, v);
end

function write_num(fn, path, name, val)
    % Write a scalar numeric/boolean attribute as a true rank-0 (H5S_SCALAR)
    % double. h5writeatt would store a scalar as a length-1 array, which the
    % numbl reader misreads (a typed array is truthy, so booleans become `true`,
    % and scalars like lineWidth/rows come back as 1-element arrays).
    fid = H5F.open(fn, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    oid = H5O.open(fid, path, 'H5P_DEFAULT');
    space = H5S.create('H5S_SCALAR');
    tid = H5T.copy('H5T_NATIVE_DOUBLE');
    try
        H5A.delete(oid, name);
    catch
    end
    aid = H5A.create(oid, name, tid, space, 'H5P_DEFAULT', 'H5P_DEFAULT');
    H5A.write(aid, 'H5ML_DEFAULT', double(val));
    H5A.close(aid);
    H5T.close(tid);
    H5S.close(space);
    H5O.close(oid);
    H5F.close(fid);
end

function write_grid(fn, path, M, dtype)
    % Write MATLAB matrix M (rows x cols) so numbl reads shape=[rows,cols] with
    % a row-major flat buffer. MATLAB h5write of an array A yields, on the C/h5
    % side, shape = fliplr(size(A)) and value = A(:). Writing M' therefore gives
    % shape = [rows, cols] and value = row-major(M). Applies equally to grids
    % (surf/imagesc/...), [N,D] vertices, and [N,3] colormap_data.
    if isempty(M); return; end
    Mt = M.';                       % [cols, rows]
    h5create(fn, path, size(Mt), 'Datatype', dtype);
    h5write(fn, path, Mt);
end

% ===========================================================================
%  Misc helpers
% ===========================================================================
function ensure_group(fn, gpath)
    fid = H5F.open(fn, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    cleaner = onCleanup(@() H5F.close(fid)); %#ok<NASGU>
    try
        gid = H5G.open(fid, gpath);
        H5G.close(gid);
        return;                     % already exists
    catch
    end
    lcpl = H5P.create('H5P_LINK_CREATE');
    H5P.set_create_intermediate_group(lcpl, 1);
    gid = H5G.create(fid, gpath, lcpl, 'H5P_DEFAULT', 'H5P_DEFAULT');
    H5G.close(gid);
    H5P.close(lcpl);
end

function write_str_attr(fn, path, name, s)
    if isempty(s); return; end
    h5writeatt(fn, path, name, s);
end

function s = text_of(h)
    % Title/XLabel/... are text objects; .String may be char, cellstr, or string.
    s = '';
    if isempty(h) || ~isprop(h, 'String'); return; end
    str = h.String;
    if isempty(str); return; end
    if iscell(str) || (isstring(str) && ~isscalar(str))
        s = char(strjoin(cellstr(str), newline));
    else
        s = char(str);
    end
end

function s = scale_string(xs, ys)
    xlog = strcmp(xs, 'log');
    ylog = strcmp(ys, 'log');
    if xlog && ylog
        s = 'loglog';
    elseif xlog
        s = 'semilogx';
    elseif ylog
        s = 'semilogy';
    else
        s = 'linear';
    end
end

function tf = axes_is_3d(ax)
    % 3-D if the view is tilted away from the default top-down [0 90], or any
    % child carries non-flat Z data.
    tf = ~isequal(round(ax.View), [0 90]);
    if tf; return; end
    for c = ax.Children(:).'
        if isprop(c, 'ZData')
            z = c.ZData;
            if ~isempty(z) && any(z(:) ~= z(1))
                tf = true; return;
            end
        end
    end
end

function lg = legend_for_axes(ax)
    lg = {};
    fig = ancestor(ax, 'figure');
    if isempty(fig); return; end
    legs = findobj(fig, 'Type', 'legend');
    for i = 1:numel(legs)
        if isequal(legs(i).Axes, ax) && ~isempty(legs(i).String)
            lg = string(cellstr(legs(i).String(:)));
            return;
        end
    end
end

function s = super_title(fig)
    s = '';
    % A sgtitle is a text object parented to the figure (Tag 'suptitle' in older
    % releases); newer MATLAB adds it via sgtitle as an axes-less Text.
    t = findobj(fig, 'Type', 'text', '-and', 'Tag', 'suptitle');
    if ~isempty(t) && ~isempty(t(1).String)
        s = text_of(t(1));
    end
end
