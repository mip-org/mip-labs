% Minimal end-to-end MatNWB example: build an NWB file, write it, read it back.
%
%   mip install --channel mip-org/labs matnwb
%   mip load matnwb

% --- build an in-memory NWB file ------------------------------------------
nwb = NwbFile( ...
    'session_description', 'a very simple recording', ...
    'identifier', 'MATNWB-EXAMPLE-0001', ...
    'session_start_time', datetime(2020, 1, 1, 12, 0, 0));

nwb.general_subject = types.core.Subject( ...
    'subject_id', 'mouse-01', 'species', 'Mus musculus');

% A 100-sample voltage trace sampled at 100 Hz.
voltage = sin(2 * pi * 5 * (0:99)' / 100);          % 5 Hz sine
ts = types.core.TimeSeries( ...
    'data', voltage, ...
    'data_unit', 'volts', ...
    'starting_time', 0.0, ...
    'starting_time_rate', 100.0);
nwb.acquisition.set('response', ts);

% --- write to disk --------------------------------------------------------
nwbFilePath = fullfile(tempdir, 'matnwb_example.nwb');
nwbExport(nwb, nwbFilePath);
fprintf('Wrote %s\n', nwbFilePath);

% --- read it back ---------------------------------------------------------
nwbIn = nwbRead(nwbFilePath, 'ignorecache');
data = nwbIn.acquisition.get('response').data.load();
fprintf('Read back %d samples for subject "%s"\n', ...
    numel(data), nwbIn.general_subject.subject_id);
