function [ contain_matrix, contain_matrix_perc ] = ...
    containingRegions( regionsA, regionsB )
%CONTAININGREGIONS checks whether regionsB are (partially) contained in regionsA.
% Returns an contain_matrix indicating partial containment of a regionA{row} 
% in regionsB{col}. contain_matrix_perc stores the percentage of containment
% (where 1 means 100% contained in another region).
% Input regions need to be stored vertice-wise (point by point).
%   
% INPUT:
%   regionsA / B - list of regions
%
% OUTPUT:
%   contain_matrix - matrix indicating whether regions overlap or not
%   contain_matrix_perc - matrix indicating contain percentage
%   
% AUTHOR: Hannes Horneber

    contain_matrix_perc = zeros(length(regionsA), length(regionsB));
    contain_matrix = zeros(size(contain_matrix_perc));
    
    for i = 1:length(regionsA)
        for j = 1:length(regionsB)
            % polygonal overlap   
            x1 = regionsA{i}(:,1);
            y1 = regionsA{i}(:,2);
            x2 = regionsB{j}(:,1);
            y2 = regionsB{j}(:,2);
            
            % move coordinates to origin
            minX = min([x1(:); x2(:)]);
            minY = min([y1(:); y2(:)]);
            x1 = x1 - minX;
            x2 = x2 - minX;
            y1 = y1 - minY;
            y2 = y2 - minY;
            
            % create object masks in nxm window
            m = max([x1(:); x2(:)]);
            n = max([y1(:); y2(:)]); 
            objMask1 = poly2mask(y1,x1,m,n);
            objMask2 = poly2mask(y2,x2,m,n);
            
            % retain: pixels of region I not contained in region J
            %retain = objMask1 & ~objMask2;
            
            % intersection (or contain): pixels of region J contained in region J
            intersection = objMask1 & objMask2;
            
            % store info
            % true if region I is partially contained in region J
            contain_matrix(i,j) = (bwarea(intersection) ~= 0);
            % perc is the percentage to which I is contained in J
            contain_matrix_perc(i,j) = bwarea(intersection) / bwarea(objMask1);
        end; clear j;
    end; clear i;
    
    % deprecated version using vectors
%     for i = 1:length(regionsA)
%         for j = 1:length(regionsB)
%             % calculate overlapRatio between bounding boxes
%             %bb_overlap = bboxOverlapRatio(objA(i).BoundingBox,objB(j).BoundingBox);
%             
%             % polygonal overlap            
%             x1 = regionsA{i}(:,1);
%             y1 = regionsA{i}(:,2);
%             x2 = regionsB{j}(:,1);
%             y2 = regionsB{j}(:,2);
%             [x_i,y_i] = polybool('intersection',x1,y1,x2,y2);
%             [x_u,y_u] = polybool('union',x1,y1,x2,y2);
%             
%             % store info
%             %intersect_matrix_geo{i, j} = [x_i,y_i];
%             intersect_matrix(i,j) = ~isempty([x_i,y_i]);
%             if ~isempty([x_i,y_i])
%                 intersect_matrix_iou(i,j) = polyarea(x_i, y_i) / polyarea(x_u, y_u);
%             else
%                 intersect_matrix_iou(i,j) = 0; 
%             end
%         end; clear j;
%     end; clear i;
    


end

