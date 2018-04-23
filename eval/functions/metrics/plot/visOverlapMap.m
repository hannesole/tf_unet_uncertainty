function [ imgOverlap ] = visOverlapMap( img1, img2 )
%VISOVERLAPMAP creates image visualizing overlapping regions of img1/img2
    R = zeros([size(img1)]);
    G = zeros([size(img1)]);
    B = zeros([size(img1)]);
    
    R(img1 == 1) = 0;
    G(img1 == 1) = 0.5;
    B(img1 == 1) = 1;
    
    R(img2 == 1) = R(img2 == 1) + 1;
    G(img2 == 1) = G(img2 == 1) + 0.2;
    B(img2 == 1) = B(img2 == 1) ;
    
    imgOverlap = cat(3, R, G, B);
    %imshow(imageRGB)
end

