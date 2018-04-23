function [ config ] = getUnetEvalConfig( netpath, AUTO_ONLY_LAST, VERBOSE )
%GETUNETEVALCONFIG Automatically parse config of Net from path (for DD/LCC)
%   get config through eval script location
%   This works for DD and LCC networks, but not for STD.
%
%   AUTO_ONLY_LAST: only the latest snapshot will be used
%   
%   Author: Hannes Horneber
%   Date-Revised: 18-May-2017 15:23:43

    % set optional variables
    if (~exist('VERBOSE', 'var'))
        VERBOSE = false;
    end
    if (~exist('AUTO_ONLY_LAST', 'var'))
        AUTO_ONLY_LAST = false;
    end

    % get config through path
    DIR_MAIN = netpath;

    [~, NETNAME, ~] = fileparts(DIR_MAIN);
    if(VERBOSE); fprintf('\n%s\n', repmat('#',1,70)); end
    if(VERBOSE); fprintf('DIR_MAIN: %s\n', DIR_MAIN); end
    if(VERBOSE); fprintf('NETNAME: %s\n', NETNAME); end
    
    NETVERSION_STR = regexp(NETNAME, '_v(\d*\_\d*)', 'tokens');
    try NETVERSION = str2double(strrep(NETVERSION_STR{1}, '_','.'));
    catch; NETVERSION = NaN; end;
    if(VERBOSE); fprintf('NETVERSION: %4.1f\n', NETVERSION); end

    cd(DIR_MAIN); % open script folder (should already be open)
    if(VERBOSE); fprintf('-- DATA --\n'); end
    try
        % IF one file containing train data path exists:
        % open default file containing path to train data
        filename_data = [NETNAME '-data_ERROR_ME.txt']; 
        PATH_DATA_TRAIN = fileread(filename_data);
        % extract DATANAME and parts of path
        [datapath, dataname_train, datafile_ext_train] = fileparts(PATH_DATA_TRAIN);
        DATANAME = strrep(dataname_train,'_train','');
        % ... and generate PATH_DATA_TEST / PATH_DATA_STATS from parts
        PATH_DATA_TEST = [datapath '/' DATANAME '_test' datafile_ext_train];
        PATH_DATA_STATS = [datapath '/' DATANAME '_stats.mat'];
    catch ME
        % IF two files for train/test data exist
        if(strcmp(ME.identifier, 'MATLAB:fileread:cannotOpenFile'))
            % assumed files containing path(s) to train/test data
            filename_data_train = [NETNAME '-data-train.txt']; 
            filename_data_test = [NETNAME '-data-test.txt']; 
            % ... if these don't work try the generic default:
            if ~(exist(filename_data_train, 'file') == 2)
                filename_data_train = 'input_files_train.txt'; 
            end;
            if ~(exist(filename_data_test, 'file') == 2)
                filename_data_test = 'input_files_test.txt'; 
            end;
            
            filenames = {filename_data_train filename_data_test};
            
            for i = 1:2 % train and test data
                filename = filenames{i};
                if(VERBOSE); fprintf('read: %s\n', filename); end
                
                if linecount(filename, true) == 1
                    % read line in file containing path to train/test data
                    if i == 1
                        PATH_DATA_TRAIN = fileread(filename);
                        % extract DATANAME and generate PATH_DATA_STATS from parts
                        [datapath, dataname_train, ~] = fileparts(PATH_DATA_TRAIN);
                        DATANAME = strrep(dataname_train,'_train','');
                        PATH_DATA_STATS = [datapath '/' DATANAME '_stats.mat'];
                    else PATH_DATA_TEST = fileread(filename); 
                    end
                else % file with multiple lines = multiple data files
                    fileID = fopen(filename);
                    % read lines in data file
                    tline = fgetl(fileID); list_idx = 1;
                    while ischar(tline)
                        % store lines in list, strtrim to remove whitespace
                        if i == 1; DATA_TRAIN{list_idx} = strtrim(tline);
                        else DATA_TEST{list_idx} = strtrim(tline); 
                        end
                        disp(tline);
                        tline = fgetl(fileID); list_idx = list_idx + 1;
                    end;
                    if(VERBOSE); fprintf('EOF: %s\n\n', filename); end
                    fclose(fileID);
                    
                    if i == 1
                        % extract DATANAME ...
                        [datapath, dataname, extension] = fileparts(DATA_TRAIN{1});
                        PATH_DATA_TRAIN = [datapath '/']; % set path to folder
                        % ... remove segmentation index ...
                        seg_idx_str = regexp([dataname extension], '(_train_\d*)\.h5' ,'tokens');
                        strrep(dataname, seg_idx_str{1}, '');
                        % ... and generate PATH_DATA_STATS from parts
                        PATH_DATA_STATS = [datapath '/' dataname '_stats.mat'];
                    else
                        [datapath, ~, ~] = fileparts(DATA_TEST{1});
                        PATH_DATA_TEST = [datapath '/']; % set path to folder
                    end
                end
            end
        else rethrow(ME);
        end 
    end
    
    % needs to be done to remove whitespace/linebreaks
    PATH_DATA_TRAIN = strtrim(PATH_DATA_TRAIN);
    PATH_DATA_TEST = strtrim(PATH_DATA_TEST);

    if(VERBOSE); fprintf('PATH_DATA_TRAIN: %s\n', PATH_DATA_TRAIN); end
    if(VERBOSE); fprintf('PATH_DATA_TEST: %s\n', PATH_DATA_TEST); end
    % check whether PATH_DATA_STATS exists, otherwise set empty
    try 
        fileread(PATH_DATA_STATS); 
    catch
        PATH_DATA_STATS = '';
    end
    if(VERBOSE); fprintf('PATH_DATA_STATS: %s\n', PATH_DATA_STATS); end

    % check normalization (net contains "ddm" keyword)
    if(~isempty(strfind(NETNAME, 'ddm')))
        % assumes normalized float input (values 0 to 1)
        NORMALIZE = false;
    else
        % assumes unnormalized int input (values 0 to 255)
        NORMALIZE = true;
    end

    % > snapshot settings
    % since Ubuntu 16.04, the old (non-hdf5) snapshots don't work anymore
    if(~isempty(strfind(NETNAME, 'lcc')) || NETVERSION >= 3.0 ...
        || NETVERSION == 0.8  || NETVERSION == 2.1 || NETVERSION == 2.6 || NETVERSION == 2.9); % nets with h5 containers
        h5_ending = '.h5'; else h5_ending = ''; 
    end;
    % get available snapshot files
    cd([DIR_MAIN '/snapshot']);
    tmp = dir(['*.caffemodel' h5_ending]);
    file_list = {tmp.name}';
    % extract iterations
    pattern = [NETNAME '_snapshot_iter_' '(\d*)' '.caffemodel' h5_ending];
    [~, tokens_iter] = regexp(file_list, pattern, 'start', 'tokens');
    % create sorted snapshot idx array, set SNAPSHOT_ITER_STEP to 1
    SNAPSHOT_IDXs = zeros(1,numel(tokens_iter));
    for nt = 1:numel(tokens_iter)
        SNAPSHOT_IDXs(nt) = str2double(tokens_iter{nt}{1});        
    end
    SNAPSHOT_IDXs = sort(SNAPSHOT_IDXs);
    if(AUTO_ONLY_LAST)
        % get last element (highest iteration)
        SNAPSHOT_IDXs = SNAPSHOT_IDXs(length(SNAPSHOT_IDXs));
    end
    SNAPSHOT_ITER_STEP = 1; % needs to be one since actual iterations are extracted
    if(VERBOSE); v = sprintf('%d ', SNAPSHOT_IDXs); fprintf('SNAPSHOT_IDXs: %s\n', v); end
    if(VERBOSE); fprintf('SNAPSHOT_ITER_STEP: %i\n', SNAPSHOT_ITER_STEP); end

    % > number of input/output classes and colormap
    if(~isempty(strfind(NETNAME, 'lcc')))
        % number of input/output classes (for stats)
        NCLASSES =      10;  % = see below
        NOUTCLASSES =	9;  % = without ignore
        % (ignore), water, grass, vegetation, ground,
        %  rock, building, road, deadwood, shadow
        CMAP_SEG = [1,1,1 ; 0,0,1 ; 0,1,0 ; ...
            0,0.502,0 ; 0.7529,0.5020,0 ; 0.5020,0.5020,0.5020 ; ...
            1,0,0 ; 1,0.5020,0 ; 1,1,0 ; 0,0,0 ]
        CONTENT = 'lcc';
    elseif(~isempty(strfind(NETNAME, 'dwlying')) || ~isempty(regexp(NETNAME, 'v\d*_\d*l')))
        % number of input/output classes (for stats)
        NCLASSES =      3;  % = bg, dw_lying, ignore
        NOUTCLASSES =	2;  % = bg, dw_lying
        CMAP_SEG = [0,0,0;140,115,215;255,255,255]/255; % bg, dw_lying, ignore
        CONTENT = 'dw_lying';
    elseif(~isempty(strfind(NETNAME, 'dwstand')) || ~isempty(regexp(NETNAME, 'v\d*_\d*s')))
        NCLASSES =      3;  % = bg, dw_standing, ignore
        NOUTCLASSES =	2;  % = bg, dw_standing
        CMAP_SEG = [0,0,0;165,215,0;255,255,255]/255; % bg, dw_standing, ignore
        CONTENT = 'dw_standing';
    else
        NCLASSES =      4;  % 4 = bg, dw_standing, dw_lying, ignore
        NOUTCLASSES =	3;  % 3 = bg, dw_standing, dw_lying (usually NCLASSES - ignore)
        CMAP_SEG = [0,0,0;165,215,0;140,115,215;255,255,255]/255; % all
        CONTENT = 'dw_combined';
    end
    if(VERBOSE); fprintf('NCLASSES: %i\n', NCLASSES); end
    if(VERBOSE); fprintf('NOUTCLASSES: %i\n', NOUTCLASSES); end
    if(VERBOSE); fprintf('CONTENT: %s\n', CONTENT); end
    
    % > separated objects or not
    % nets with obj separation functionality
    if(~isempty(strfind(NETNAME, 'lcc')) || ...
        NETVERSION <= 3.2 || ...
        NETVERSION >= 3.6 || ...
        NETVERSION == 0.8); 
        OBJ_SEP = false;
    else 
        OBJ_SEP = true;
    end;
    if(VERBOSE); fprintf('OBJ_SEP: %s\n', num2str(OBJ_SEP)); end
    
    % save configuration for output
    config.DIR_MAIN = DIR_MAIN;
    config.NETNAME = NETNAME;
    config.NETVERSION = NETVERSION;
    
    config.PATH_DATA_TRAIN = PATH_DATA_TRAIN;
    config.PATH_DATA_TEST = PATH_DATA_TEST;
    config.PATH_DATA_STATS = PATH_DATA_STATS;
    
    if(exist('DATA_TRAIN', 'var')) 
        config.DATA_TRAIN = DATA_TRAIN;
    else config.DATA_TRAIN = [];
    end
    if(exist('DATA_TEST', 'var')) 
        config.DATA_TEST = DATA_TEST;
    else config.DATA_TEST = [];
    end
    
    config.NORMALIZE = NORMALIZE;
    
    config.SNAPSHOT_IDXs = SNAPSHOT_IDXs;
    config.SNAPSHOT_ITER_STEP = SNAPSHOT_ITER_STEP;
    
    config.NCLASSES = NCLASSES;
    config.NOUTCLASSES = NOUTCLASSES;
    config.CMAP_SEG = CMAP_SEG;
    config.CONTENT = CONTENT;
    config.OBJ_SEP = OBJ_SEP;
    
end
 
