function [ ] = relabelFolder( DIR_SRC_labels, DIR_TARGET_relabeled, relabel_matrix, ALLOW_FAST_RELABEL, CUSTOM_CMAP, VERBOSE )
%RELABEL relabel masks in a folder according to relabel_matrix
%{
    Loads masks at location DIR_SRC_labels, relabels them using a given
    relabel_matrix and writes relabeled masks to DIR_TARGET_relabeled.

	INPUT
		DIR_SRC_labels          -   source folder
		DIR_TARGET_relabeled 	-	target folder
        relabel_matrix          -   matrix containing replacement info
		ALLOW_FAST_RELABEL      -	true for simple replacements, false if
                                    replacements are consecutive 
                                    (e.g. x -> 2, 2 -> x)
        CUSTOM_CMAP             -   custom color map for relabeled images
                                    otherwise old one will be used (labels 
                                    reassigned, but colors not)

	OUTPUT
		no output. relabeled masks are written to DIR_TARGET_relabeled

    AUTHOR: Hannes Horneber

	CHANGELOG:
		20-Oct-2016 14:38:38: first version
        16-Nov-2016 16:02:32: fixed bug, added CUSTOM_CMAP
%}
    % set optional variables
    if (~exist('VERBOSE', 'var'))
        VERBOSE = false;
    end
    if (~exist('ALLOW_FAST_RELABEL', 'var'))
        ALLOW_FAST_RELABEL = false;
    end
    
    % get nr of files for loop
    cd(DIR_SRC_labels);
    tmp = dir('*.tif');
    files = {tmp.name}';
    nFiles = length(files);
    
    if(VERBOSE) 
        fprintf(['\nRELABELING ' int2str(nFiles) ' files ' ...
            '\n to ' DIR_TARGET_relabeled '\n']); 
    end
    
    % create dir if not exists
    mkdir(DIR_TARGET_relabeled);
    
    for i = 1:nFiles 
        if(VERBOSE) fprintf('.'); end
    
        % load mask
        cd(DIR_SRC_labels);
        mask_file = files{i};
        if (~exist('CUSTOM_CMAP', 'var'))
            [mask,cmap] = imread(mask_file); % load mask with color index
        else
        	mask = imread(mask_file);
            cmap = CUSTOM_CMAP;
        end
        
        if(VERBOSE) 
            %imshow(mask, cmap); % display with color map
            fprintf('.'); 
            %linebreak after each 10 images
            if(mod(i, 10) == 0)
                fprintf('\n'); 
            end
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

        % write relabeled mask
        cd(DIR_TARGET_relabeled);
        imwrite(mask, cmap, mask_file);
    end

    if(VERBOSE) fprintf('\n done. \n\n'); end

end

