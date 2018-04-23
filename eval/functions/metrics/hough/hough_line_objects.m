function [ obj, obj_map, contain_matrix_perc ] = hough_line_objects( img, varargin )
%HOUGH_LINE_OBJECTS returns regions corresponding to a hough neighborhood
% Uses a Hough transform to detect lines in the image. The strongest line
% shapes are reconstructed and returned as objects.
%
% INPUT:
%   img         - input img to work with
%   MODE        - 
%
% NAME/VALUE PAIRS (with default values):
%     'MODE', 'default', ...      % method with which objects are extracted
%                                 knownModes: {'threshold (= default)','extended_max'};
%                                 watershed mode not yet implemented
%     'MAX_OVERLAP', 0.8, ...     % of two objects overlapping more than this the smaller one is removed
%     'THRESH_NSTEPS', 15, ...    % threshold param: nr of steps in thresh scale
%     'THRESHOLD', 6, ...         % threshold param: threshold where to cut
%     'THRESH_MINSIZE', 8, ...    % threshold param: min size of Hough cluster
%     'VISUALIZE', false, ...     % will show a panel with map, hough and obj.%
%
% OUTPUT:
%   obj, obj_map, contain_matrix_perc
%
% CHANGES:
%   19-Jan-2017 12:51:12 - first version
%   09-May-2017 23:53:50 - refinements, modes, more testing, objects
%   17-May-2017 18:39:30 - varargin, refinements, extended_max
%   
% AUTHOR: Hannes Horneber
    
        
%% SETTINGS
opts = struct( ...                      %# define defaults
    'MODE', 'default', ...      % method with which objects are extracted
    'MAX_OVERLAP', 0.8, ...     % of two objects overlapping more than this the smaller one is removed
    'THRESH_NSTEPS', 15, ...    % threshold param: nr of steps in thresh scale
    'THRESHOLD', 6, ...         % threshold param: threshold where to cut
    'THRESH_MINSIZE', 8, ...    % threshold param: min size of Hough cluster
    'VISUALIZE', false, ...     % will show a panel with map, hough and obj.
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
end; clear inpName pair;
    
    %% Hough Transform
    [H, theta, rho] = hough(img);  % visualize with: figure, imagesc(H);
       
    if strcmp(opts.MODE, 'watershed') 
        % Custom watershed with refined merging
        try   
            stepsize = -1;
            maxHeight = max(H(:));
            minHeight = min(H(:)); % may set this higher to ignore lowest base / ground

            mask = zeros(size(img));
            for height = maxHeight : stepsize : minHeight % descend in height
                mask(img >= height) = true; % get pixels of this height
                %{
                % IDEA:
                % [peak, peak_num] = bwlabel(mask);

                % for each new region assign peak idx
                % grow each peak when scanning over height and store properties
                    % which pixel at which height (implicit)
                    % peak_max (highest height in peak) (implicit)

                % when two regions/peaks X and Y merge:
                % calc corresponding stats and merge if certain criteria are met
                % assume X >= Y (set X to the higher peak (earlier found))
                    % h-min:	minimal depth of valley between peaks X and Y
                    %           is the steps that both peaks were "visible" before 
                    %           they merge 
                    %
                    % pk-diff:	difference in height 
                    %           (always >= 0 for peak X - peak Y)
                    % 
                    % pk-dist:  distance between peaks
                    %           calculate (volume/height weighted) center of
                    %           gravity of each peak X and Y and get distance
                    %           between those.
                    %
                    % distance, difference and h-min together can be combined
                    % to create a merge criteria for the separatedness between
                    % peaks X and Y

                % another idea/addition:
                % stop growing a peak criteria
                    % once a region has reached a relative peakedness, stop
                    % expanding this region and ignore the connected base in
                    % further iterations
                    %
                    % e.g. if a relative height is reached, if it is clear that
                    % the mountain is tall and the furthermore contributing
                    % base is part of a mountain range that contributes to
                    % several peaks (then ignore this base mountain range)
                %}
            end
        catch 
        end; % end try
    elseif strcmp(opts.MODE, 'extended_max') 
        H1 = imgaussfilt(H,1);                          % blur image
        % figure, imagesc(H1); % visualize
        H2 = imtophat(H1, strel('disk',5));             % tophat transform
        % figure, imagesc(H2); % visualize
        H3 = imextendedmax(H2, 10);                     % extended maximums
        % figure, imagesc(H3); % visualize
        
        s = regionprops(H3,'Centroid','PixelIdxList','Area','BoundingBox');  % Extract each blob
        % figure,imagesc(H),axis image
        M = false(size(H));                             % init label map
        
        for i = 1:numel(s)
            %x = ceil(s(i).Centroid);
            tmp = H1*0;
            tmp(s(i).PixelIdxList) = 1;
            tmp2 = tmp.*H2;

            % The maximum amplitude and location
            [refV,b] = max(tmp2(:));
            [x2,y2] = ind2sub(size(H),b);

            % select the region around local max amplitude    
            tmp = bwselect(H2>refV*0.6,y2,x2,4);
            
            s(i).Sum = sum(tmp(:));
            M(tmp) = true;
                
            %[xi, yi] = find(tmp);
            %hold on, plot(yi,xi,'r.')
            %hold on, text(y2+10,x2,num2str(i),'Color','white','FontSize',16)    
        end
        [L, L_num] = bwlabel(M);   
        
    else % default: if strcmp(opts.MODE, 'threshold')
        % Thresholding Hough Space
        % result is a label map, assigning a Hough region to an object
        
        thresh = multithresh(H, opts.THRESH_NSTEPS); % create threshold steps
        q_image = imquantize(H, thresh);              % quantize image

        q_image(q_image <= opts.THRESHOLD) = 0;       % regions under threshold are thrown away
        q_image(q_image > opts.THRESHOLD) = 1;        % ... while all others are preserved
        q_image = imbinarize(q_image);
      
        B = bwareaopen(q_image, opts.THRESH_MINSIZE); % Filter really small regions
        [L, L_num] = bwlabel(B);                      % Label connected components
    end
       
    %% create objects from labeled Hough regions
    obj_pool(L_num, 1).id = [];     % preallocate struct for object candidates
    obj_mask = cell([L_num 1]);     % preallocate obj_mask cell
    
    % overlap map is only used for visualization
    if opts.VISUALIZE; overlap = zeros(size(img)); end
    
    % Reconstruct the line shapes and create objects
    % each r/c value corresponds to a Hough line parameter (rho/theta)
    for m = 1:L_num
        [r, c] = find(L(:,:) == m);
        if ~isempty(r)
            segmented_im = hough_bin_pixels(img, theta, rho, [r(1) c(1)]);
            for i = 1:size(r(:))
                seg_part = hough_bin_pixels(img, theta, rho, [r(i) c(i)]);
                segmented_im(seg_part==1) = 1;
            end
            img_line_object = segmented_im; % figure; imshow(img_line_object);
            
            % keep only biggest continous region
            % this avoids that fragments of other branches in the same "line" as
            % the line shaped object are assigned to it
            img_line_object = bwareafilt(img_line_object, 1);
            
            % get obj attributes:
            %   Area, BoundingBox, id, boundaries
            obj_props = regionprops(img_line_object,'BoundingBox','Area');
            obj_boundary = bwboundaries(img_line_object, 'noholes');
            
            % create struct with obj attributes: 
            obj_pool(m).Area = obj_props.Area;
            obj_pool(m).BoundingBox = obj_props.BoundingBox;
            obj_pool(m).id = m;
            obj_pool(m).boundary = obj_boundary{1};
            
            obj_mask{m} = img_line_object;  % store object mask for when objects are filtered
            %obj_map(img_line_object) = m;  % create map later with kept objects 
            
            % mark pixels in overlap map
            if opts.VISUALIZE; overlap(img_line_object) = overlap(img_line_object) + 1; end
        end
    end
    
    %% select objects and create obj_map
    % reject redundant objects (some objects are completely contained in
    % others and should be rejected)
    
    % smallest objects first
    [~,ord] = sort([obj_pool.Area], 'ascend');
    obj_pool = obj_pool(ord);
    obj_mask = obj_mask(ord);
    % this ensures that deletes are done in the right order (when iterating 
    % through the objects, we never delete one that might contain other
    % smaller regions (since those are always handled first).
    
    if opts.VISUALIZE; obj_pool_flagDel = false(numel(obj_pool),1); end
    
    obj_map = zeros(size(img));     % init obj_map
    contain_matrix_perc = [];       % init overlap matrix
    % apply selection only if more than 1 object was found
    if(L_num > 1)       
        [~, contain_matrix_perc] = ...
            containingRegions({obj_pool.boundary}, {obj_pool.boundary});
        nKeepObj = 0;
        for i = 1:L_num
            if max(contain_matrix_perc(i, 1:end ~= i)) > opts.MAX_OVERLAP
                % object overlapping more than MAX_OVERLAP with another object
                % are not added to keep list
                contain_matrix_perc(1:end,i) = 0; % remove from contain matrix
                contain_matrix_perc(i,1:end) = 0; % remove from contain matrix
                if opts.VISUALIZE; obj_pool_flagDel(i) = true; end
            else
                % keep the object, update ID
                nKeepObj = nKeepObj + 1;
                obj_pool(i).id = nKeepObj;
               
                % append to kept objects
                if ~exist('obj','var') || isempty(obj)
                    obj = obj_pool(i);
                else
                    obj = [obj; obj_pool(i)];
                    %obj(end + 1) = obj(i);
                end
                
                % overlapping regions are overwritten by the latest obj
                obj_map(obj_mask{i}) = obj_pool(i).id;
            end
        end
    else
        % if only one object was found, simply return it and it's obj_mask
        obj = obj_pool;
        obj_map(obj_mask{1}) = 1;
    end
    
    %% PLOT results
    if opts.VISUALIZE
        try            
            % requires an open and active figure before function call
            
            % Fig. 1: plot image
            subplot(1,5,1);
            imshow(img);
            title('img');
            hold on;

            % Fig. 2: overlap
            subplot(1,5,2);
            imshow(label2rgb(overlap, prism));
            title('line objects');
            hold on;

            % Fig. 3: line objects
            % show detected img objects with boundaries
            subplot(1,5,3);
            imshow(img); hold on;
            %colors = prism(10);
            colors = lines(10);
            for j=1:length(obj_pool)
                cidx = mod(j-1,size(colors, 1))+1;
                boundary = obj_pool(j).boundary;
                
                plot(boundary(:,2), boundary(:,1),...
                    'Color', colors(cidx, :),'LineWidth',2);

                % randomize text position for better visibility
                rndRow = ceil(length(boundary)/(mod(rand,7)+1));
                col = boundary(rndRow,2); row = boundary(rndRow,1);
                if obj_pool_flagDel(j)
                    displaytxt = ['(' num2str(j) ')'];
                else
                    displaytxt = num2str(j);
                end
                h = text(col+1, row-1, displaytxt);
                set(h,'Color',colors(cidx, :),'FontSize',12);
            end; clear col row rndRow h;
    %         imshow(label2rgb(img_line_objects, prism));
            title('');
            hold on;

            % Fig. 4: plot Hough
            subplot(1,5,4);
    %         imshow(H,[],'XData',theta,'YData',rho);
    %         xlabel('\theta'), ylabel('\rho');
    %         axis on, axis normal;
            H_gray = mat2gray(H);
            imshow(H_gray);
            title('Hough transform');

            % Fig. 4: plot Hough
            subplot(1,5,5);
            imshow(label2rgb(L, lines));
            title('Hough neighborhoods');
            hold on;
        catch ME
            warning(['Error while visualizing: ' ME.message]);
        end; % end try
    end

end

