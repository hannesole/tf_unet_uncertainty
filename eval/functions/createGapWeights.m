function [ weights_gap, weights_simple ] = createGapWeights( im_instancelabels )
%CREATEGAPWEIGHTS creates weightmap with large weights between objects
%
% This function generates a weightmap "weights_gap" for given
% instancelabels. Weights in this map are highest (>> 1) in gaps between 
% objects, low (< 1) for background and 1 for objects themselves.
% "weights_simple" is a byproduct, simply containing an object map:
%   1 for object, 0 for background / ignore
%
% This was extracted from the script 'finetune_2dcellnet'.
%
% Labelling conventions for input (im_instancelabels):
% 0	-	background
% 1	-	ignore: use this for image regions(pixels) that you want to ignore.
%                 This is useful for example when
%                 -you don't want to label all objects in your image
%                 -for regions that are ambiguous
%                 -regions that are out of focus 
% >=2 -	instancelabels for fg objects / cells: Same label can be used for multiple cells,
%         however touching cells must have pairwise different instancelabels.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %parameters for weight generation - don't change unlesss you know what you're doing
    wanted_element_size_um = 0.5;

    sigma1_um = 5;
    sigma1_px = sigma1_um / wanted_element_size_um;
    foregroundBackgroundRatio = 0.1;

    borderWeightFactor = 50;
    borderWeightSigma_um = 3; 
    sigma2_px= borderWeightSigma_um / wanted_element_size_um;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %create "extra weights" based on instancelabels
    labels = zeros(size(im_instancelabels));
    extraweights = zeros(size(im_instancelabels));
    extraweights2 = zeros(size(im_instancelabels));
    
    % init ignore_mask
    ignore_mask = im2bw(zeros(size(im_instancelabels)));
    ignore_mask(im_instancelabels == 1) = true;

    sDisk = strel('disk',1);
    bordermask = zeros(size(im_instancelabels));

    inds = unique(im_instancelabels(:));
    inds(inds == 0) = []; %remove bg
    inds(inds == 1) = []; %remove ignore 

    for li = 1:length(inds)
        mask = (im_instancelabels== inds(li));
        bordermask = bordermask + (imdilate(mask,sDisk));%touching borders > 1
    end

    mask2 = single(ismember(im_instancelabels, inds));
    mask2(bordermask>1) = 0;

    labels = mask2;

    min1dist = 1e10*ones(size(im_instancelabels)); % --> dists for gaussian border weight decay
    min2dist = 1e10*ones(size(im_instancelabels)); % --> dists for touching
    for li = 1:length(inds)

        mask = (im_instancelabels==inds(li));
        d = bwdist(mask);
        min2dist = min(min2dist,d);
        newMin1  = min( min1dist, min2dist);
        %here the magic happens: if its not already down == close to one instance,
        %it gets raised again.
        newMin2  = max( min1dist, min2dist); 
        min1dist = newMin1;
        min2dist = newMin2;
    end

    va = 1 - foregroundBackgroundRatio;
    wa = exp( -(min1dist.^2)/(2*sigma1_px.^2));
    we = exp( -(min1dist+min2dist).^2/sigma2_px.^2);

    extraweights = borderWeightFactor*we;
    extraweights(labels>0) = 0; 

    extraweights2 = va*wa;
    extraweights2(labels>0) = 0; %--> std. decaying border weights, look good

    weights_gap = zeros(size(im_instancelabels),'single');
    weights_gap(labels > 0) = 1;

    weights_gap(labels == 0) = extraweights(labels == 0) + extraweights2(labels == 0) + foregroundBackgroundRatio;
    weights_gap(ignore_mask) = 0;

    weights_simple = zeros(size(im_instancelabels),'single');
    weights_simple(labels > 0) = 1;
    weights_simple(labels == 0) = foregroundBackgroundRatio;
    weights_simple(ignore_mask) = 0;
    %End  weight extravaganza
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

