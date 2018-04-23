function [ obj, obj_labelmap ] = im2obj( im_bw, varargin )
%IM2OBJ from bw image generate list of obj with some attributes 
%
%   INPUT
%       im_ml           - input image (multi labeled image)
%
%   NAME/VALUE PAIRS:
%       MODE            - knownModes: {'watershed','mi_labels','houghlines','default'};
%       LABEL_BG        - default: 0; % label of background class
%       MIN_AREA        - default: 1; % objects below this size are rejected
%       MAX_OVERLAP     - of two objects overlapping more than this the smaller one is removed
%       VISUALIZE       - will trigger figures to be created when using houghlines mode
%
%   Various modes availabe:
%       
%       DEFAULT: CONNECTED COMPONENT LABELING
%           regionprops / bwlabel are used to create objects and labels
%
%       WATERSHED SEGMENTATION
%           MODE = 'watershed';
%           a distance transform with (h-minima) watershed segmentation
%           creates objects and labels
%
%       HOUGH LINE TRANSFORMATION AND SEGMENTATION; 
%           MODE = 'houghline';
%           the hough line transform uses connected components in the image
%           as a basis for applying hough transformations and extracting
%           line shaped (!) objects. 
%           
%           WARNING 1: Objects may be smaller than they appeared before as
%           they are rebuild based on the corresponding Hough peak. Some
%           areas of the object may be missing.
%
%           WARNING 2: The method allows for overlapping objects! The 
%           resulting obj_labelmap may not fully show all objects
%           (overlapped regions will be masked by "higher" objects, where
%           "higher" means: later detected).
%   
%       MULTIINSTANCE LABELING 
%           MODE = 'mi_labels';
%           if multiinstance labels are available, these can be used to
%           cleanly separate and create objects. For this, LABEL_BG needs
%           to be specified (or will by default be set to zero).
%   
%   created attributes are
%       id          - an ID assigned to the object
%       Area        - area of the object
%       BoundingBox - enclosing region
%       boundary	- the objects outline in pixel coordinates
%
% CHANGES: 
%   12-Mar-2017 04:21:12 - first comprehensive version
%   17-May-2017 17:59:23 - houghline transform and watershed added
%   21-May-2017 00:47:36 - bugfixes
%   
% AUTHOR: Hannes Horneber

%% SETTINGS
opts = struct( ...                      %# define defaults
    'MODE', 'default', ...      % method with which objects are extracted
    'LABEL_BG', 0, ...          % background label (needed for mi_labels)
    'MIN_AREA', 30, ...         % objects below this size are rejected
    'MAX_OVERLAP', 0.8, ...     % of two objects overlapping more than this the smaller one is removed
    ...                         % e.g. used in houghlines
    'VISUALIZE', false, ...     % only used for visualizing Houghlines
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

%% OBJECT EXTRACTION
% select mode and go. Default mode (bottom) just goes with connected component labeling.

    if strcmp(opts.MODE, 'watershed')
        %% WATERSHED SEGMENTATION
        D = bwdist(~im_bw);         % distance transform
        D = -D;                     % complement the distance transform
        D(im_bw == opts.LABEL_BG) = Inf;     % ... force background pixels to Inf D(~im_bw) = Inf;
        
                                    % remove shallow local minima
        D = imhmin(D,2);            % 2 seems a good height threshold 
        obj_labelmap = watershed(D);
        obj_labelmap(im_bw == opts.LABEL_BG) = 0;	% clip background regions
      
        % get obj regions and boundaries from labelmap
        obj_all = regionprops(obj_labelmap,'BoundingBox','Area','PixelIdxList');
        obj_boundaries = bwboundaries(obj_labelmap, 'noholes');
                
        % the order in which obj_boundaries are stored
        % does NOT correspond to the labels -> create index
        obj_boundaries_idx = zeros(numel(obj_boundaries), 1);
        for k=1:length(obj_boundaries)
            col = obj_boundaries{k}(1,2);
            row = obj_boundaries{k}(1,1);
            obj_boundaries_idx(k) = obj_labelmap(row,col);
        end; clear k;
        
        % create return struct with objects
        keepObjects = 0; % init         % keep objects that satisfy MIN_AREA
        obj = repmat( ...       
            struct('id',[],'Area',[],'BoundingBox',[],'boundary',[]), ... 
            numel(obj_boundaries), 1 ); % preallocate array of structs
        for i = 1:length(obj_all)
            if obj_all(i).Area > opts.MIN_AREA
                % keep object: add to obj struct
                keepObjects = keepObjects + 1;
                obj(keepObjects).id = keepObjects;
                obj(keepObjects).Area = obj_all(i).Area;
                obj(keepObjects).BoundingBox = obj_all(i).BoundingBox;  
                obj(keepObjects).boundary = obj_boundaries{obj_boundaries_idx==i};
                % ... and change obj_labelmap
                if(i ~= keepObjects) % not necessary if i == keepObjects
                    obj_labelmap(obj_labelmap == i) = keepObjects;
                end
            else
                % reject object: delete from obj_labelmap
                obj_labelmap(obj_labelmap == i) = 0;
            end
        end; clear i;
        % delete preallocated empty rows
        if keepObjects < numel(obj)
            obj(keepObjects+1 : numel(obj_boundaries)) = [];
        end
        
    elseif strcmp(opts.MODE, 'houghlines')
        %% HOUGH LINE TRANSFORMATION AND SEGMENTATION
        % generate connected components 
        [~,labels] = bwboundaries(im_bw, 'noholes');            % get labels
        bb_props = regionprops(labels,'BoundingBox','Area');    % get BoundingBoxes
        bb_img = cell(length(bb_props), 1);                     % init array

        % get image regions with (potentially multiple) objects
        for i = 1:length(bb_props)                        
            bb_idx = bb_props(i).BoundingBox + [0.5 0.5 -1 -1] ;
            subregion = labels( ...                 % get bb content from labels
                   bb_idx(2):(bb_idx(2)+bb_idx(4)), ...
                   bb_idx(1):(bb_idx(1)+bb_idx(3)));
            subregion(~ismember(subregion, i)) = 0; % remove all labels that belong to other bb  
            bb_img{i} = subregion;                  % store image
        end; clear i bb_coords subregion;

        % opts.VISUALIZE the individual regions
        if opts.VISUALIZE;
            figure('Name', 'object candidate regions');
            for i=1:length(bb_img)
                subplot(10,ceil(length(bb_img)/10),i);
                imshow(bb_img{i});
            end
        end
        
        % Hough Line Object split
        % split each connected component into likely objects
        obj_labelmap = zeros(size(im_bw));    % 
        if(opts.VISUALIZE); fig1 = figure('Name', 'Hough Line Object split'); end
        for i = 1:length(bb_img) % i = 16, 32, 104
            if bb_props(i).Area > opts.MIN_AREA;
                im_bb = bb_img{i};
                %, im_bw); %bb_props(i).BoundingBox + [0.5 0.5 -1 -1];
                
                % create line shaped objects
                [ obj_bb, obj_bb_map, ~ ] = ...
                    hough_line_objects(im_bb, 'MODE', 'extended_max', ...
                    'VISUALIZE', opts.VISUALIZE, 'MAX_OVERLAP', opts.MAX_OVERLAP);
                
                % objects were created in their own bounding box space,
                % hence coordinates are relative to that. Add bounding box
                % coordinates to transform coords to global coords:
                for j = 1:numel(obj_bb)
                    % subtract 0.5 bbox offset that is redundantly 
                    % generated when creating each obj 
                    obj_bb(j).BoundingBox = obj_bb(j).BoundingBox + [... 
                        bb_props(i).BoundingBox(1)-0.5 ... 
                        bb_props(i).BoundingBox(2)-0.5 ...
                        0 0];
                    obj_bb(j).boundary(:,1) = obj_bb(j).boundary(:,1) + bb_props(i).BoundingBox(2)-0.5;
                    obj_bb(j).boundary(:,2) = obj_bb(j).boundary(:,2) + bb_props(i).BoundingBox(1)-0.5;
                end
                
                % append to obj: struct with all objects
                if ~exist('obj','var') || isempty(obj)
                    obj = obj_bb;              	% initialize object container
                else
                    offset = length(obj);   	% offset IDs and labelmap
                    for j = 1:length(obj_bb); obj_bb(j).id = obj_bb(j).id + offset; end
                    obj_bb_map(obj_bb_map ~= 0) = obj_bb_map(obj_bb_map ~= 0) + offset;
                    obj = [obj; obj_bb];        % append obj to global obj store
                end

                obj_labelmap_bak = obj_labelmap;
                obj_labelmap = obj_labelmap_bak;
                % append labelmap into global labelmap
                bb_idx = bb_props(i).BoundingBox + [0.5 0.5 -1 -1];    % convert bbox to indices             
                occupation_map = obj_labelmap( ...              % store existing labels
                        bb_idx(2):(bb_idx(2)+bb_idx(4)), ...    % in obj_labelmap bbox area
                        bb_idx(1):(bb_idx(1)+bb_idx(3)));
                obj_bb_map(occupation_map ~= 0) = ...           % combine with new labels
                    occupation_map(occupation_map ~= 0);
               
                obj_labelmap( ...                               % insert patch into map
                        bb_idx(2):(bb_idx(2)+bb_idx(4)), ...
                        bb_idx(1):(bb_idx(1)+bb_idx(3))) = obj_bb_map;
                
                if(opts.VISUALIZE); waitforbuttonpress; end
                if(opts.VISUALIZE); clf(fig1,'reset'); end
            end
        end
        
        if ~exist('obj','var')
            % return empty struct/labelmap if no objects are found
            obj = repmat( ...       
                struct('id',[],'Area',[],'BoundingBox',[],'boundary',[]), ... 
                0, 1 );
            obj_labelmap = zeros(size(im_bw));
        end
    elseif strcmp(opts.MODE,'mi_labels')
        %% MULTIINSTANCE LABELING
        % instances are separated based on existing labeling
        % for each label bwconncomp / bwlabel selects all objects      
        if(opts.VERBOSE); fprintf('  separate instances '); end
        
        % get all labels
        labels = unique(im_bw(:));
        labels = labels(labels ~= opts.LABEL_BG); % rm bg label from list
        
        if ~isempty(labels)
            for j = 1:length(labels)
                label_mask = (im_bw == labels(j));
                %imshow(label_mask); waitforbuttonpress;

                [obj_l, obj_l_labelmap] = im2obj(label_mask); 
                % plotObjects( label_mask, obj_l, 'id');
                % figure, imagesc(objLabel_map);

                if ~exist('obj','var') || isempty(obj)
                    obj = obj_l;                    % initialize object container
                    obj_labelmap = obj_l_labelmap;  % initialize obj_labelmap
                else
                    offset = length(obj);
                    % offset IDs and offset labelmap
                    for i = 1:length(obj_l); obj_l(i).id = obj_l(i).id + offset; end
                    obj_l_labelmap(obj_l_labelmap ~= 0) = ...
                        obj_l_labelmap(obj_l_labelmap ~= 0) + offset;

                    if ~length(obj_l) == 0
                        % append labelmap to global labelmap
                        obj_labelmap(obj_l_labelmap ~= 0) = ... 
                            obj_l_labelmap(obj_l_labelmap ~= 0);

                        % append obj to global obj store
                        obj = [obj; obj_l];
                    end
                end

                if(opts.VERBOSE); fprintf('.'); end
            end; if(opts.VERBOSE); fprintf(' done.\n'); end
        else
            % return empty struct/labelmap if no objects are found
            obj = repmat( ...       
                struct('id',[],'Area',[],'BoundingBox',[],'boundary',[]), ... 
                0, 1 );
            obj_labelmap = zeros(size(im_bw));
        end
        
    else
        %% DEFAULT: CONNECTED COMPONENT LABELING
        % get obj regions and boundaries from bw_image (implicit bwconncomp)
        obj_all = regionprops(im_bw,'BoundingBox','Area');
        [obj_boundaries, obj_labelmap] = bwboundaries(im_bw, 'noholes');
        
        % create return struct with MIN_AREA objects
        keepObjects = 0; % init         % keep objects that satisfy MIN_AREA
        obj = repmat( ...       
            struct('id',[],'Area',[],'BoundingBox',[],'boundary',[]), ... 
            numel(obj_boundaries), 1 ); % preallocate array of structs
        for i = 1:length(obj_all)
            if obj_all(i).Area > opts.MIN_AREA
                % keep object: add to obj struct
                keepObjects = keepObjects + 1;
                obj(keepObjects).id = keepObjects;
                obj(keepObjects).Area = obj_all(i).Area;
                obj(keepObjects).BoundingBox = obj_all(i).BoundingBox;  
                obj(keepObjects).boundary = obj_boundaries{i};
                % ... and change obj_labelmap
                if(i ~= keepObjects) % not necessary if i == keepObjects
                    obj_labelmap(obj_labelmap == i) = keepObjects;
                end
            else
                % reject object: delete from obj_labelmap
                obj_labelmap(obj_labelmap == i) = 0;
            end
        end; clear i;
        % delete preallocated empty rows
        if keepObjects < numel(obj)
            obj(keepObjects+1 : numel(obj_boundaries)) = [];
        end
    end
    
end

