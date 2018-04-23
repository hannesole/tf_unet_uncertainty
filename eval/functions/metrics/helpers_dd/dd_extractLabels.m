function [ im_out ] = dd_extractLabels( im_ml, varargin) 
%DD_EXTRACTLABELS Summary of this function goes here
%
%   INPUT
%       im_ml           - input image (multi labeled image)
%
%   NAME/VALUE PAIRS:
%       EXTRACT_MODE    - knownModes: {'mi_dwstand','mi_dwlying','discard_ignore'};
%       REMOVE_NOISE    - default: false
%                           remove small holes / artifacts
%       NOISE_SIZE      - default: 42
%       BW_OUT          - default: false
%                           return only binary image (if multiple labels are
%                           to be extracted, then the image is "flattened" 
%                           before return.
%       LABELS_EXTRACT  - default: 1; % labels to be extracted
%       LABEL_BG        - default: 0; % label of background class
%       
%   OUTPUT
%       im_out          - image only with selected labels

%% SETTINGS
opts = struct( ...                      %# define defaults
    'EXTRACT_MODE', 'default', ...
    'LABELS_EXTRACT', 1, ...    % label(s) that are extracted: int (array)
    'LABEL_BG', 0, ...          % background label
    'BW_OUT', false, ...        % flatten to b/w image (bg=false, labels=true)
    'REMOVE_NOISE', false, ...	% remove noise in input image
    'NOISE_SIZE', 42, ...       % max size of holes/noise that are removed
    'VERBOSE', false ...    
);
%# override options defaults if propertyName/propertyValue pairs are given
optionNames = fieldnames(opts);         %# read acceptable option names
nArgs = length(varargin);               %# count arguments
if round(nArgs/2)~=nArgs/2; error('Function needs propertyName/propertyValue pairs'); end
for pair = reshape(varargin,2,[])       %# pair is {propName;propValue}
   inpName = upper(pair{1});            %# make case insensitive
   if any(strcmp(inpName,optionNames)); opts.(inpName) = pair{2};
   else error('%s is not a recognized parameter name',inpName); end
end
    %% MODE SETTINGS
    % label specific settings
    % 13 = ignore?      | 12
    % 12 = bg           | 11
    % 7 - 11 = dwlying	| 6-10
    % 1 - 6 = dwstand 	| 0-5
    %opts.LABEL_BG = 11; % background in src labels
    %opts.LABEL_IGNORE = 12; % ignore in src labels
   
    if strcmp('mi_dwstand',opts.EXTRACT_MODE)
        opts.LABELS_EXTRACT = 0:5;   % dwstanding labels in src
        opts.LABEL_BG = 11;          % background label in src
    elseif strcmp('mi_dwlying',opts.EXTRACT_MODE)
        opts.LABELS_EXTRACT = 6:10;  % dwlying labels in src
        opts.LABEL_BG = 11;          % background label in src
    elseif strcmp('discard_ignore',opts.EXTRACT_MODE)
        % rm last label from list
        labels = unique(im_ml(:));
        opts.LABELS_EXTRACT = labels(labels ~= max(labels(:)));
    else
        warning('Using default extract mode.');
    end


    %% %%%%%%%%%%%%%%%%%%%%%
    % SELECT CLASS %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%

    % select labels that should be extracted
    selection_mask = false(size(im_ml));
    selection_mask(ismember(im_ml,opts.LABELS_EXTRACT)) = true;
 
    % this is the image with the selected relevant regions
    im_sel = uint8(ones(size(im_ml))) * opts.LABEL_BG; % set all to bg
    im_sel(selection_mask) = im_ml(selection_mask);
    %imshow(selection_mask);

    %% %%%%%%%%%%%%%%%%%%%%%
    % REMOVE NOISE %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%

    if(opts.REMOVE_NOISE)
        labels = unique(im_sel(:));
        labels = labels(labels ~= opts.LABEL_BG); % rm bg label from list

        if(opts.VERBOSE); fprintf('  remove noise '); end
        im_nf = im_sel;
        for j = 1:length(labels)
            label_mask = (im_nf == labels(j));
            % remove small objects / artifacts ("white noise")
            label_mask_nf = bwareaopen(label_mask, opts.NOISE_SIZE, 4);
            noise_mask = label_mask ~= label_mask_nf;
            % set detected white noise to background
            im_nf(noise_mask) = opts.LABEL_BG;

            % remove small holes ("black noise")
            label_mask_neg = ~label_mask;
            label_mask_neg_nf = bwareaopen(label_mask_neg, opts.NOISE_SIZE, 4);
            noise_mask = label_mask_neg ~= label_mask_neg_nf;
            % set detected black noise to background
            im_nf(noise_mask) = labels(j);

            if(opts.VERBOSE); fprintf('.'); end
        end; if(opts.VERBOSE); fprintf(' done.\n'); end
        % imshow(im_nf, cmap); waitforbuttonpress;
    else im_nf = im_sel;
    end
    
    %% %%%%%%%%%%%%%%%%%%%%%
    % FILTER % OUTPUT %%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    if(opts.BW_OUT)
        im_out = false(size(im_nf));
        im_out(im_nf ~= opts.LABEL_BG) = true;
    else
        % ma
        im_out = im_nf;
    end

end

