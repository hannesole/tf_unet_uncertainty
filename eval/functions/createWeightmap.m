function [ weightmap ] = createWeightmap( mask, weight_matrix, EXTRAVAGANZA )
%createWeightmap Generate weight map from mask and weight matrix
%
% WEIGHT MATRIX holds the weight assignment in the form: 
%   [ [class1, weight]; [class2, weight]; ... ]
% respectively with contour weights: 
%   [ [class1, weight, weight_contour]; [class2, ... ] ... ]
%
% EXTRAVAGANZA = true
% will trigger GapWeights to be created. Requires a segmentation mask with
% gaps between objects, otherwise this makes little sense.


    if (~exist('EXTRAVAGANZA', 'var'))
        EXTRAVAGANZA = false;
    end

    if(~EXTRAVAGANZA)
        [nClasses, dimWeights] = size(weight_matrix);
        if(dimWeights == 3) CONTOUR = true;
        else CONTOUR = false; end;

        shape = size(mask);
        weightmap = double(ones(shape)); % init weight 1
        if(~CONTOUR)
            % fast replacement using matrix operation
            for class = 1:nClasses
                if(weight_matrix(class, 2) ~= 1) % we don't need to reset ones
                    % set all weights for class to corresponding weight value
                    weightmap(mask == weight_matrix(class, 1)) = weight_matrix(class, 2);
                end
            end
        else
            contour = mask - imerode(mask, ones(5,5)); % create contour
            % slow: iterate over mask pixels and set weights accordingly
            for x = 1:shape(1)
                for y = 1:shape(2)
                    % find value in relabel matrix and assign new value
                    idx = find(weight_matrix(:, 1) == mask(x,y));
                    if(idx) % classes without entry in the weight matrix won't be set
                        if(contour(x,y)); mask(x,y) = weight_matrix(idx, 3); 
                        else mask(x,y) = weight_matrix(idx, 2); end
                    end
                end                           
            end
        end
    else
        im_instancelabels = createInstancelabels(mask);
        [weightmap, ~] = createGapWeights(im_instancelabels);
    end
    %fprintf('');
end

