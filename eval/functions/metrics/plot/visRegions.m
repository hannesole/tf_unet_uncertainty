function [ RGB ] = visRegions( binary_image )
%VISREGIONS creates image with colorized connected components
%   https://de.mathworks.com/help/images/ref/label2rgb.html
    
    CC = bwconncomp(binary_image);
    % Convert the label matrix into RGB image, using default settings.
    L = labelmatrix(CC); 
    RGB = label2rgb(L, 'lines');
    figure, imshow(RGB);
end

