function [ confusionMatrix, confusionMatrix_aug, confusionMatrix_per, confusionMatrix_per2 ] = confMatrix( groundtruth, result, nClasses )
%STATS: Calculates the confmatrix over segmentation/classification
%{
	scans given segmenation, compares to groundtruth and calculates:
    - the confusion matrix
    - additional stats (TP, FP, TN, FN, IoU, ...) 

	INPUT
		groundtruth     -   ...
		result          -	...
		nClasses        -	number of classes

	OUTPUT
		confusionMatrix         - plain confusion matrix (incl. totals)
        confusionMatrix_aug     - more infos (TP, FP, TN, FN, IoU, ...) 
        confusionMatrix_per     - recall matrix
        confusionMatrix_per2    - precision matrix

    AUTHOR: Hannes Horneber

	CHANGELOG:
		27-Sep-2016 12:41:39: first version
%}

fprintf('\nCompute ConfusionMatrix');
% initialize empty arrays
confusionMatrix = zeros(nClasses+1, nClasses+1);
confusionMatrix_per = zeros(nClasses+1, nClasses+1);
confusionMatrix_per2 = zeros(nClasses+1, nClasses);

% augmented confusionMatrix (with additional stats)
offsetCol = 8;
confusionMatrix_aug = zeros(nClasses+1, nClasses+offsetCol);
%stats_idx
%(1,c): IoU
%(2,c): intersect
%(3,c): union
%(4,c) - (7,c): TP, FP, TN, FN
%(8,c): total actual class pixels (1-6)
%(c,1): total precited class pixels (1-6)
%colNames = {'IoU', 'I', 'U', 'TP', 'FP', 'TN', 'FN'};

% calc Matrix
imgSize = size(groundtruth);
nr_pixels = imgSize(1) * imgSize(2);

for y = 1:imgSize(1)
    for x = 1:imgSize(2)
        if ~(groundtruth(y,x) > nClasses)
            confusionMatrix(groundtruth(y,x)+1, result(y,x)+1) = ...
                confusionMatrix(groundtruth(y,x)+1, result(y,x)+1) + 1;

            % count total groundtruth pixels per class
            confusionMatrix(1, result(y,x)+1) = ...
                confusionMatrix(1, result(y,x)+1) + 1;

            % count total prediction pixels per class
            confusionMatrix(groundtruth(y,x)+1, 1) = ...
                confusionMatrix(groundtruth(y,x)+1, 1) + 1;



            % AUGMENTED MATRIX
            confusionMatrix_aug(groundtruth(y,x)+1, result(y,x)+offsetCol) = ...
                confusionMatrix_aug(groundtruth(y,x)+1, result(y,x)+offsetCol) + 1;

            % count total groundtruth pixels per class
            confusionMatrix_aug(1, result(y,x)+offsetCol) = ...
                confusionMatrix_aug(1, result(y,x)+offsetCol) + 1;

            % count total prediction pixels per class
            confusionMatrix_aug(groundtruth(y,x)+1, offsetCol) = ...
                confusionMatrix_aug(groundtruth(y,x)+1, offsetCol) + 1;

            for desired_class = 1:nClasses
                intersect(y,x) = (groundtruth(y,x) == desired_class) && (result(y,x) == desired_class);
                union(y,x) = (groundtruth(y,x) == desired_class) || (result(y,x) == desired_class);

                if(union(y,x))
                    confusionMatrix_aug(groundtruth(y,x)+1, 3) = ...
                        confusionMatrix_aug(groundtruth(y,x)+1, 3) + 1;
                end
                if(intersect(y,x))
                    confusionMatrix_aug(groundtruth(y,x)+1, 2) = ...
                        confusionMatrix_aug(groundtruth(y,x)+1, 2) + 1;
                end


                % get stats and create colormap of TP, FP, TN, FN           
                if(union(y,x) && intersect(y,x))
                    % true positive
                    confusionMatrix_aug(groundtruth(y,x)+1, 4) = ...
                        confusionMatrix_aug(groundtruth(y,x)+1, 4) + 1;
                elseif(result(y,x) == desired_class)
                    % false positive
                    confusionMatrix_aug(groundtruth(y,x)+1, 5) = ...
                        confusionMatrix_aug(groundtruth(y,x)+1, 5) + 1;
                elseif(groundtruth(y,x) == desired_class)
                    % false negative
                    confusionMatrix_aug(groundtruth(y,x)+1, 7) = ...
                        confusionMatrix_aug(groundtruth(y,x)+1, 7) + 1;
                else
                    % true negative
                    confusionMatrix_aug(groundtruth(y,x)+1, 6) = ...
                        confusionMatrix_aug(groundtruth(y,x)+1, 6) + 1;
                end
            end
        else 
            nr_pixels = nr_pixels - 1;
        end

    end                    
end

for desired_class = 1:nClasses
    %IoU
    if(confusionMatrix_aug(desired_class+1, 3) == 0) % catch division by zero
        confusionMatrix_aug(desired_class+1, 1) = 100;
    else confusionMatrix_aug(desired_class+1, 1) = ...
            round((confusionMatrix_aug(desired_class+1, 2) / ...
            confusionMatrix_aug(desired_class+1, 3)) * 10000) / 100;
    end

    %percentages1
    if confusionMatrix(desired_class+1, 1) ~= 0 % catch division by zero
        for desired_class2 = 1:nClasses
            confusionMatrix_per(desired_class+1, 1) = ...
                confusionMatrix(desired_class+1, 1) / nr_pixels;
            confusionMatrix_per(desired_class+1, desired_class2+1) = ...
                 round((confusionMatrix(desired_class+1, desired_class2+1) / ...
                    confusionMatrix(desired_class+1, 1)) * 10000) / 100;
        end
    end

    %percentages2
    if confusionMatrix(1, desired_class+1) ~= 0 % catch division by zero
        for desired_class2 = 1:nClasses
            confusionMatrix_per2(1, desired_class) = ... 
                confusionMatrix(1, desired_class+1) / nr_pixels;
            confusionMatrix_per2(desired_class2+1, desired_class) = ...
                 round((confusionMatrix(desired_class2+1, desired_class+1) / ...
                    confusionMatrix(1, desired_class+1)) * 10000) / 100;
        end
    end
end

confusionMatrix_aug = horzcat(confusionMatrix_aug(:,offsetCol:offsetCol+nClasses), ...
    confusionMatrix_aug(:,1:offsetCol-1));
fprintf('... done\n');

end

