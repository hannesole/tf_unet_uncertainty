function [ label_info, col_names ] = evalLabels( files, DIR_SRC_labels, VERBOSE, nClasses)
%DATAINFO Gets info about provided segmentation files
%
% scans given files and calculates:
% - pixelcount of each segmentation class (area covered by class)
% - classratio of each segmentation class (percentage covered by class)
% - objcount of each segmentation class (count distinct regions of segmented objects)
% 
% INPUT
% -----
% files             -   list of files at DIR_SRC_labels
% DIR_SRC_labels    -	directory of files
% nClasses          -   number of (continous!) different labels
% VERBOSE           -	text output true/false
% 
% OUTPUT
% -----
% struct containing info about the given tiles
% 
% CHANGELOG:
% ----------
% 25-Jul-2016 15:32:09: first version
%
% AUTHOR: Hannes Horneber

%% init
if(~exist('nClasses', 'var'))
    nClasses = 4;
end

class(nClasses-1).stats = []; % temp for each iteration, no bg class (=> -1)
class(nClasses-1).centro = []; % temp for each iteration, no bg class (=> -1)
% allocate
nFiles=length(files);
classpixelcount = zeros(nFiles, nClasses);
classratio = zeros(nFiles, nClasses);
classobjcount = zeros(nFiles, nClasses);

%% get info about data
for i = 1:nFiles
    % load mask file
    cd(DIR_SRC_labels);
    [mask,cmap] = imread(files{i}); % load mask with color index
    if(VERBOSE) fprintf('\n%s', files{i}); end

    % allocate
    mask_shape = size(mask);
    
    % this is only needed for plotting
    %imshow(mask, cmap); % display with color map
    %hold on

    % IMAGE ANALYSIS
    for n = 1:nClasses % start with 2 means ignore background class
        m = n-1; % classes in mask begin with 0
        mask_class = (mask == m);
        % sum pixels per class
        classpixelcount(i, n) = sum(mask_class(:));
        classratio(i,n) = classpixelcount(i, n) / (mask_shape(1) * mask_shape(2));
        % find closed regions in binary mask of classevalLabels
        temp_stats = regionprops(mask_class, 'Centroid', 'Area');
        % count number of objects (closed regions)
        if(~isempty(temp_stats)) % class has pixels
            classobjcount(i, n) = length(temp_stats);
        end

        % this is only needed if regionprops are processed (plotting etc.)
        %{
        % store properties
        class(n).stats = temp_stats;

        % show centroids of closed regions
        class(n).centro = vertcat(temp_stats.Centroid);
        if(~isempty(class(n).centro)) % class has regions
            scatter(class(n).centro(:,1),class(n).centro(:,2),40,'filled');
        end
        %}
        if(VERBOSE) fprintf('.'); end
    end
    %contour = mask - imerode(mask, ones(5,5));
end
if(VERBOSE) fprintf('\n'); end

% create cell array from collected info
label_info = horzcat(files, num2cell(classpixelcount), num2cell(classratio), num2cell(classobjcount));

% define column names
col_names = cell(1, size(label_info, 2)); % get nr of columns
col_name1 = 'filename';
col_name2 = 'npx_c';
col_name3 = 'rtio_c';
col_name4 = 'nobj_c';

col_names{1} = col_name1;
for i = 2:(1+nClasses)
    col_names{i} = [col_name2 num2str(i-1)]; % 2:5
end
for i = (2+nClasses):(1+nClasses*2)
    col_names{i} = [col_name3 num2str(i-(1+nClasses))]; % 6:9
end
for i = (2+nClasses*2):(1+nClasses*3)
    col_names{i} = [col_name4 num2str(i-(1+nClasses*2))]; % 10:13
end

%% print summary
if(VERBOSE) 
    fprintf('________________________\n'); 
    fprintf('\nSUMMARY OF DATA INFO:\n');
    for i = 1:nFiles
        fprintf([files{i} '\t' strrep([num2str(cell2mat(label_info(i,[(2+nClasses):(1+nClasses*2)])),'%.2f\t') '\t' num2str(cell2mat(label_info(i,[(2+nClasses*2):(1+nClasses*3)])),'%u\t') '\n'], '.', ',')]);
    end
    fprintf('________________________\n');
end

end

