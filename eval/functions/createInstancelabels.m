function [ im_instancelabels ] = createInstancelabels( im_classlabel_sep, TARGET_LABEL_BG )
%CREATEINSTANCELABELS take class labels and create instance labels
%
% Input are label maps with segmented objects that can be separated with
% bwconncomp. This function outputs instance labels that conform to the
% input requirements of createGapWeights (bg_label = 0, ignore_label = 1).
% 
% AUTHOR: Hannes Horneber
% 
% CHANGES:
%     01-Feb-2017 21:59:59 - first version

    if (~exist('TARGET_LABEL_BG', 'var'))
        TARGET_LABEL_BG = 0;
    end

    % create instancelabels
    obj_mask = im_classlabel_sep ~= TARGET_LABEL_BG;
    im_instancelabels = bwlabel(obj_mask, 4);

    % arrange for bg label = 0, ignore label = 1
    im_instancelabels = im_instancelabels + 1;
    im_instancelabels(im_instancelabels == 1) = 0;

end

