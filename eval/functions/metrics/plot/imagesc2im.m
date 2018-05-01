function [ output_img ] = imagesc2im( input_img, cmap )
%IMAGESC2IM generates an RGB img of a gray value map scaled with colors
%   the resulting image can be displayed using imshow()
%   this replaces use of imagesc (which creates a figure, not an image)
%   pass cmap = 0 for default heatmap
    if cmap == 0
        cmap = jet(256);
    end
    output_img = ind2rgb(ceil(size(cmap,1)*input_img), cmap);
end

