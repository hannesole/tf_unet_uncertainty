function [ mask ] = denoise(mask, denoise_matrix, TRESHOLD)
%DENOISE removes small unconnected areas in a multiclass label mask
% Extends removal of small objects to non-binary images. 
% The matlab function <a href="https://de.mathworks.com/help/images/ref/
% bwareaopen.html">bwareaopen()</a> only works on binary images.
%
% mask = DENOISE(mask, denoise_matrix) call with default treshold (40).
% mask = DENOISE(mask, denoise_matrix, TRESHOLD)
%
% Input
% -----
% mask              -   the mask that needs denoising
% denoise_matrix    -   noise pixels need to be replaced. The denoise
%                       matrix contains information on which "colors" 
%                       (a class of pixel values) are assumed to be 
%                       noisy and into which color noise pixels of the 
%                       corresponding noise will be converted 
%                       (e.g. denoise_matrix = [[1 0]; [2 0]];
%                       "noise of color 1" -> set 0, 
%                       "noise of color 2" -> set 0).
% TRESHOLD          -   nr of connected pixels below which an area is
%                       considered noise. see doc of bwareopen(). 
%                       40 is a good value for small areas.
% 
% Output
% ------
% mask  -   clean mask, in which noise of classes in denoise_matrix
%           is removed (and replaced with classes specified in 
%           denoise_matrix)
%
% 
% Changelog
% ---------
% 20-Oct-2016 14:38:38: first version
% 
% Author: Hannes Horneber
%
% See also BWAREAOPEN.

    % initialize optional variables
    if (~exist('TRESHOLD', 'var'))
        TRESHOLD = 40;
    end

    [nr_of_replacements, ~] = size(denoise_matrix);
    for replacement_nr = 1:nr_of_replacements 
        class_label = denoise_matrix(replacement_nr, 1);
        
        binary_mask = (mask == class_label);
        binary_mask_clean = bwareaopen(binary_mask, TRESHOLD );
        noise = (binary_mask - binary_mask_clean);
        mask(noise == 1) = denoise_matrix(replacement_nr, 2);
    end
end

