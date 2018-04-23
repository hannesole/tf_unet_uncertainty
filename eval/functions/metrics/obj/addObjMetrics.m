function [ objA ] = addObjMetrics( objA, objB, USE_EACH_OBJ_ONCE)
%ADDOBJMETRICS adds IoU and overlaps to objA (for overlapping regions of objB)
%   ...
    if ~exist('USE_EACH_OBJ_ONCE','var'); USE_EACH_OBJ_ONCE = true; end
    
    % check whether inputs contain objects
    if isempty(objA)
        objA.overlaps = [];
        objA.iou = [];
    elseif ~isempty(objA) && isempty(objB)
            for i=1:length(objA)
                objA(i).overlaps = [];
                objA(i).iou = 0;
            end
    else
        % get intersection matrix to determine overlapping regions
        [~, intersect_matrix_iou] = ...
            intersectingRegions({objA.boundary}, {objB.boundary}, ...
                false, cell2mat({objA.BoundingBox}'), cell2mat({objB.BoundingBox}'));

        % using each object only once is standard
        if(USE_EACH_OBJ_ONCE)
            for i=1:length(objA)
                % identify highest iou amongst overlaps
                max_iou = max(intersect_matrix_iou(i,:));
                if max_iou > 0
                    % take prediction_object with highest overlap as match
                    match = find(intersect_matrix_iou(i,:) == max_iou);

                    % remove matched objects from intersect_matrix (to avoid further matches)
                    intersect_matrix_iou(i,:) = 0;        % remove groundtruth_object
                    intersect_matrix_iou(:,match) = 0;    % remove prediction_object

                    % save match and corresponding iou
                    objA(i).overlaps = match;
                    objA(i).iou = max_iou;
                else
                    % no match found
                    objA(i).overlaps = [];
                    objA(i).iou = 0;
                end
            end
        else
            % this will give all overlapping objects
            for i=1:length(objA)
                % get list of all overlapping OBJ in objB and store them
                objA(i).overlaps = find(intersect_matrix_iou(i,:));
                if numel(objA(i).overlaps) == 1
                    objA(i).iou = intersect_matrix_iou(objA(i).overlaps);
                elseif isempty(numel(objA(i).overlaps))
                    objA(i).iou = 0;
                else
                    % if more than one object overlaps, IoU is undefined
                    objA(i).iou = NaN;
                end
            end
        end
    end


    
    
    
    
end

