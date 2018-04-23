function [ IoU, intersect, union ] = iouPerClass( groundtruth, result, desired_class )
%IOU: intersect over union per class (pixelwise)
%{
    Calculates the intersect over union (IoU) based on correctly classified
	pixels in the image (not objectwise)

	INPUT
		groundtruth		-	img ground truth
		result          -	img result
		desired_class	-	class which to observe

	OUTPUT
		IoU (ratio)

    AUTHOR: Hannes Horneber

	CHANGELOG:
		25-Jul-2016 14:31:50 - first version
%}

    % initialize empty arrays
    intersect = zeros(size(groundtruth));
    union = zeros(size(groundtruth));

    % get intersect and union
    imgSize = size(groundtruth);
    for x = 1:imgSize(1)
        for y = 1:imgSize(2)
            intersect(x,y) = (groundtruth(x,y) == desired_class) && (result(x,y) == desired_class);
            union(x,y) = (groundtruth(x,y) == desired_class) || (result(x,y) == desired_class);
        end                    
    end
    
    % count pixels
    intersect_sum = sum(intersect(:));
    union_sum = sum(union(:));
        
    % catch division by zero
    if(union_sum == 0)
        if(intersect_sum == 0)
            IoU = 1.0;
        else IoU = 0.0;
        end
    else IoU = intersect_sum / union_sum;
    end
    
end

