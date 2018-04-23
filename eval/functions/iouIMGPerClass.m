function [ IoU, rgb, stats, stats_fpc ] = iouIMGPerClass( groundtruth, result, desired_class )
%IOU: intersect over union per class (pixelwise) with IMG
%{
    Calculates the intersect over union (IoU) based on correctly classified
	pixels in the image (not objectwise)

	INPUT
		groundtruth		-	img ground truth
		result          -	img result
		desired_class	-	class which to observe

	OUTPUT
		IoU (ratio)     -   the IoU on that class
        rgb             -   rgb image colorized according to IoU
        stats           -   vector containing pixelcount of TP, FP, TN, FN
                            (true/false positives/negatives)
        stats_fpc       -   pixelcount of FP vs. class 2/3
                            (eg for class 2 this is class 3 and vice versa)
    
    AUTHOR: Hannes Horneber

	CHANGELOG:
		25-Jul-2016 14:31:50 - first version
        26-Jul-2016 18:04:05 - added rgb image
        29-Jul-2016 17:11:52 - added stats (TP, FP, TN, FN)
        21-Apr-2018 17:59:14 - adjusted to bg class = 0, class1 = 1, ...
%}

    % initialize empty arrays
    intersect = zeros(size(groundtruth));
    union = zeros(size(groundtruth));
    rgb = zeros([size(groundtruth) 3]);
    
    % TP, FP, TN, FN
    stats = [0,0,0,0];
    % false positive of THE other non-bg class (eg class 2 this is class 3)
    stats_fpc = 0;
    
    % get intersect and union
    imgSize = size(groundtruth);
    for x = 1:imgSize(1)
        for y = 1:imgSize(2)
            % get intersect and union
            intersect(x,y) = (groundtruth(x,y) == desired_class) && (result(x,y) == desired_class);
            union(x,y) = (groundtruth(x,y) == desired_class) || (result(x,y) == desired_class);
            
            % get stats and create colormap of TP, FP, TN, FN           
            if(desired_class == 0)
                if(union(x,y) && intersect(x,y))
                    % true positive
                    stats = stats + [1,0,0,0];
                elseif(result(x,y) == desired_class)
                    % false positive
                    stats = stats + [0,1,0,0];
                elseif(groundtruth(x,y) == desired_class)
                    % false negative
                    stats = stats + [0,0,0,1];
                else
                    % true negative
                    stats = stats + [0,0,1,0];
                end
            elseif(desired_class == 1)
                if(union(x,y) && intersect(x,y))
                    % true positive
                    stats = stats + [1,0,0,0];
                    rgb(x,y,:) = [165,215,0]/255;
                elseif(result(x,y) == desired_class)
                    % false positive 
                    stats = stats + [0,1,0,0];
                    % false positive
                    if(groundtruth(x,y) == 3)
                        % vs. dw_ly label
                        rgb(x,y,:) = [180,0,0]/255;
                        stats_fpc = stats_fpc + 1;
                        %fprintf('\nfalse positive vs label 3 %i, %i> %i, %i, %i', x, y, rgb(x,y,1), rgb(x,y,2), rgb(x,y,3));
                    else
                        % vs. background label
                        rgb(x,y,:) = [255,215,0]/255;
                    end
                elseif(groundtruth(x,y) == desired_class)
                    % false negative
                    stats = stats + [0,0,0,1];
                    if(result(x,y) == 2)
                        % vs. dw_ly label
                        rgb(x,y,:) = [90,0,0]/255;
                    else
                        % vs. background label
                        rgb(x,y,:) = [90,135,0]/255;
                    end
                else
                    % true negative
                    stats = stats + [0,0,1,0];
                end
            end
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

