function [ intersect_matrix, intersect_matrix_iou ] = ...
    intersectingRegions( regionsA, regionsB, CONVEX, bboxA, bboxB )
%INTERSECTINGREGIONS checks whether regionsB overlap with regionsA.
% Returns an intersect matrix indicating overlaps of a regionA{row} with a
% regionsB{col}. intersect_matrix_geo stores the overlapping polygon.
% Input regions need to be stored vertice-wise (point by point).
%   
% INPUT:
%   regionsA / B    - list of N regions
%   CONVEX          - (optional) if regions are strictly convex,
%                       computation will be faster. By default false.
%   bboxA / B       - (optional) list of N bounding boxes (as mat array 
%                       with rows [X, Y, width, height]). If none are
%                       provided, bounding boxes are created internally for
%                       faster computation.
%
% OUTPUT:
%   intersect_matrix - matrix indicating whether regions overlap or not
%   intersect_matrix_area - matrix indicating overlap area
%   
% AUTHOR: Hannes Horneber

    if ~exist('CONVEX','var'); CONVEX = false; end
    if ~exist('bboxA','var') || ~exist('bboxB','var')
        % get enclosing rectangle for each polygon
        poly2rect = @(c) [min(c(:,1)),min(c(:,2)),...
            1+max(c(:,1))-min(c(:,1)),1+max(c(:,2))-min(c(:,2))];
        bboxA = cell2mat(cellfun(poly2rect,regionsA,'UniformOutput',0)');
        bboxB = cell2mat(cellfun(poly2rect,regionsB,'UniformOutput',0)');
    else

    % compute rectangle intersections
    % rectint(bboxA,bboxB) is just as fast...
    bbox_intersect = bboxOverlapRatio(bboxA,bboxB); 
    
    intersect_matrix_iou = zeros(length(regionsA), length(regionsB));
    intersect_matrix = zeros(size(intersect_matrix_iou));
    
    for i = 1:length(regionsA)
        for j = 1:length(regionsB)
            % only calculate intersect if bounding boxes intersect
            if bbox_intersect(i,j) > 0  
                % get region coordinates to build polygons   
                x1 = regionsA{i}(:,1);
                y1 = regionsA{i}(:,2);
                x2 = regionsB{j}(:,1);
                y2 = regionsB{j}(:,2);
                
                % use (faster) polygon based computation for convex regions
                if CONVEX
                    % move coordinates to origin
                    % this reduces size of the window created by poly2mask
                    % and will save memory and computational time
                    minX = min([x1(:); x2(:)]);
                    minY = min([y1(:); y2(:)]);
                    x1 = x1 - minX;
                    x2 = x2 - minX;
                    y1 = y1 - minY;
                    y2 = y2 - minY;

                    % create object masks in window of size n x m
                    m = max([x1(:); x2(:)]);
                    n = max([y1(:); y2(:)]); 
                    objMask1 = poly2mask(y1,x1,m,n);
                    objMask2 = poly2mask(y2,x2,m,n);
                    
                    % compute intersect and union
                    intersection = objMask1 & objMask2;
                    union = objMask1 | objMask2;

                    % store info
                    intersect_matrix(i,j) = (bwarea(intersection) ~= 0); % avoid division by zero
                    if (bwarea(union) ~= 0)
                        intersect_matrix_iou(i,j) = bwarea(intersection) / bwarea(union);
                    else
                        intersect_matrix_iou(i,j) = 0; % avoid division by zero
                    end
                else
                    % compute intersect and union
                    [x_i,y_i] = polybool('intersection',x1,y1,x2,y2);
                    [x_u,y_u] = polybool('union',x1,y1,x2,y2);

                    % store info
                    intersect_matrix(i,j) = ~isempty([x_i,y_i]);
                    if ~isempty([x_i,y_i]) && ~isempty([x_u,y_u]) 
                        intersect_matrix_iou(i,j) = polyarea(x_i, y_i) / polyarea(x_u, y_u);
                    else
                        intersect_matrix_iou(i,j) = 0; 
                    end
                end
            else
                intersect_matrix(i,j) = false;
                intersect_matrix_iou(i,j) = 0;
            end
        end; clear j;
    end; clear i;
    
end

