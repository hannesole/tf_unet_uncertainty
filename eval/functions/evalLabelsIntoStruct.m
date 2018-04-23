function [ label_info ] = evalLabelsIntoStruct( files, DIR_SRC_labels, VERBOSE )
%DATAINFO Gets info about provided segmentation files
%{
	scans given files and calculates:
    - pixelcount of each segmentation class (area covered by class)
    - classratio of each segmentation class (percentage covered by class)
    - objcount of each segmentation class (count distinct regions of segmented objects)

	INPUT
		files           -   list of files at DIR_SRC_labels
		DIR_SRC_labels	-	directory of files
		VERBOSE         -	text output true/false

	OUTPUT
		struct containing info about the given tiles

    AUTHOR: Hannes Horneber

	CHANGELOG:
		25-Jul-2016 13:23:07: first version
        25-Jul-2016 15:32:09: deprecated (branched cell array version)
%}

%% init
% allocate
nFiles=length(files);
label_info(nFiles).name = [];
label_info(nFiles).pixelcount = [];
label_info(nFiles).classratio = [];
label_info(nFiles).objcount = [];
nClasses = 4; % max(mask(:))
class(nClasses-1).stats = []; % temp for each iteration
class(nClasses-1).centro = []; % temp for each iteration

%% get info about data
for i = 1:nFiles
    % load mask file
    cd(DIR_SRC_labels);
    [mask,cmap] = imread(files{i}); % load mask with color index
    if(VERBOSE) fprintf('\n%s', files{i}); end

    % allocate
    mask_shape = size(mask);
    pixelCount = zeros(1, nClasses);
    objCount = zeros(1, nClasses);

    % this is only needed for plotting
    %imshow(mask, cmap); % display with color map
    %hold on

    % IMAGE ANALYSIS
    for n = 1:nClasses % start with 2, ignore background class
        m = n-1; % classes begin with 0
        mask_class = (mask == m);
        % sum pixels per class
        pixelCount(n) = sum(mask_class(:));
        % find closed regions in binary mask of class
        temp_stats = regionprops(mask_class, 'Centroid', 'Area');
        % count number of objects (closed regions)
        if(~isempty(temp_stats)) % class has pixels
            objCount(n) = length(temp_stats);
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

    % store
    label_info(i).name = files{i};
    label_info(i).pixelcount = pixelCount;
    label_info(i).classratio = pixelCount / (mask_shape(1) * mask_shape(2));
    label_info(i).objcount = objCount;
end % label_info is now filled and can be returned
if(VERBOSE) fprintf('\n'); end

%% print summary
if(VERBOSE) 
    fprintf('________________________\n'); 
    fprintf('\nSUMMARY OF DATA INFO:\n');
    for i = 1:nFiles
        fprintf([files{i} '\t' strrep([num2str(label_info(i).classratio,'%.2f\t') '\t' num2str(label_info(i).objcount,'%u\t') '\n'], '.', ',')]);
    end
    fprintf('________________________\n');
end

end

