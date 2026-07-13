% Test script for matnwb (mip-org/labs).
%
% Exercises MatNWB's core purpose — building an in-memory NWB file, writing it
% to disk with nwbExport, and reading it back with nwbRead — using only base
% MATLAB (no add-on toolboxes) so it runs on the channel's base-only CI. Also
% confirms the bundled +types classes are pre-generated (no code-generation
% step needed) and that external_packages/fastsearch is on the path.
%
% matnwb ships no MEX, so there is no MEX-coverage gate here.
rng('default');

% --- pre-generated types are available ------------------------------------
% matnwb generates the +types.* classes from the NWB schema. This release
% ships them pre-generated for its active schema version; confirm that.
activeSchema = matnwb.common.getActiveSchemaVersion();
assert(~isempty(activeSchema), ...
    'No active NWB schema version — +types classes were not shipped generated');
fprintf('Active NWB schema version: %s\n', activeSchema);

% --- external_packages/fastsearch is on the path --------------------------
% +util/loadTimeSeriesData depends on this bare (non-namespaced) function.
assert(~isempty(which('fastsearch')), ...
    'fastsearch not on path — external_packages/fastsearch is missing');
assert(fastsearch((0:9)', 4.2, 1) == 6, 'fastsearch returned unexpected index');

% --- build an in-memory NWB file ------------------------------------------
fprintf('Building an NwbFile...\n');
nwb = NwbFile( ...
    'session_description', 'matnwb mip package test', ...
    'identifier', 'MIP-MATNWB-0001', ...
    'session_start_time', datetime(2020, 1, 1, 12, 0, 0));

nwb.general_subject = types.core.Subject( ...
    'subject_id', 'mouse-01', 'species', 'Mus musculus');

data = (1:100)';
timestamps = (0:99)' * 0.01;
ts = types.core.TimeSeries( ...
    'data', data, 'data_unit', 'volts', 'timestamps', timestamps);
nwb.acquisition.set('test_timeseries', ts);

% --- write to disk --------------------------------------------------------
outDir = tempname; mkdir(outDir);
cleanupOut = onCleanup(@() rmdir(outDir, 's'));
nwbFilePath = fullfile(outDir, 'test.nwb');

fprintf('Writing %s ...\n', nwbFilePath);
nwbExport(nwb, nwbFilePath);
assert(isfile(nwbFilePath), 'nwbExport did not produce a file');

% --- read it back ---------------------------------------------------------
% Pass an explicit, writable savedir so the read does not depend on the
% install directory being writable (nwbRead regenerates the in-memory
% namespace from the file's embedded schema — see README).
saveDir = tempname; mkdir(saveDir);
cleanupSave = onCleanup(@() rmdir(saveDir, 's'));

fprintf('Reading back with nwbRead...\n');
nwbIn = nwbRead(nwbFilePath, 'savedir', saveDir);

assert(strcmp(nwbIn.identifier, 'MIP-MATNWB-0001'), 'identifier mismatch');
assert(strcmp(nwbIn.session_description, 'matnwb mip package test'), ...
    'session_description mismatch');
assert(strcmp(nwbIn.general_subject.subject_id, 'mouse-01'), ...
    'subject_id mismatch');

readData = nwbIn.acquisition.get('test_timeseries').data.load();
assert(isequal(readData(:), data), 'TimeSeries data did not round-trip');

readTimestamps = nwbIn.acquisition.get('test_timeseries').timestamps.load();
assert(max(abs(readTimestamps(:) - timestamps)) < 1e-12, ...
    'TimeSeries timestamps did not round-trip');

% --- read again with ignorecache (no class re-generation) -----------------
% The file was written with the active schema version, so its classes are
% already on the path; 'ignorecache' skips regeneration entirely.
fprintf('Reading back with nwbRead ''ignorecache''...\n');
nwbIn2 = nwbRead(nwbFilePath, 'ignorecache');
readData2 = nwbIn2.acquisition.get('test_timeseries').data.load();
assert(isequal(readData2(:), data), 'ignorecache read did not round-trip');

fprintf('SUCCESS\n');
