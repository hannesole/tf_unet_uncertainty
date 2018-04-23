function [ mask ]  = relabel( mask, relabel_matrix, ALLOW_FAST_RELABEL )
%RELABEL relabel masks according to relabel_matrix
% Loads masks at location DIR_SRC_labels, relabels them using a given
% relabel_matrix and writes relabeled masks to DIR_TARGET_relabeled.
% 
% INPUT
% mask                -   mask to relabel
% relabel_matrix      -   matrix containing replacement info
% ALLOW_FAST_RELABEL  -	true for simple replacements, false if
%                         replacements are consecutive 
%                         (e.g. x -> 2, 2 -> x)
% 
% OUTPUT
% mask                -   relabeled mask
% 
% AUTHOR: Hannes Horneber
% 
% CHANGELOG:
% 20-Oct-2016 14:38:38: first version

    % set optional variables
    if (~exist('ALLOW_FAST_RELABEL', 'var'))
        ALLOW_FAST_RELABEL = false;
    end
    
    if(ALLOW_FAST_RELABEL)
        % fast replacement using matrix operation
        % doesn't work if labels are changed consecutively (x -> 2, 2 -> x)
        % either change order or use slow replacement (pixelwise)
        [nr_of_replacements, ~] = size(relabel_matrix);
        for replacement_nr = 1:nr_of_replacements 
            mask(mask == relabel_matrix(replacement_nr, 1)) = relabel_matrix(replacement_nr, 2);
        end
    else
        % slow replacement
        % iterate over mask and apply values according to relabel_matrix matrix
        shape = size(mask);
        for x = 1:shape(1)
            for y = 1:shape(2)
                % find value in relabel matrix and assign new value
                idx = find(relabel_matrix(:, 1) == mask(x,y));
                if(idx) mask(x,y) = relabel_matrix(idx, 2); end
            end
        end
    end
end

