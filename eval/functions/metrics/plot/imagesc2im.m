function [ output_img ] = imagesc2im( input_img, cmap )
%IMAGESC2IM generates an RGB img of a gray value map scaled with colors
%   the resulting image can be displayed using imshow()
%   this replaces use of imagesc
    output_img = ind2rgb(ceil(size(cmap,1)*input_img), cmap);
end

