% STD UNET EVALUATION
%
% This script can be used to evaluate results from a segmentation net.
% Input are the (groundtruth) labels, the prediction w/ softmax and 
% and optionally uncertainty map. 
% 
% Core functionality encompasses: 
%   - Generate statistics / graphs / confusion matrix for results
%   - work with previously generated results
%  
% CHANGES: 
%   18-Apr-2017 11:47:47	- branched from std_eval_v3_7_x2.m (caffe net eval)
%
% AUTHOR: Hannes Horneber

clear all
%% Initialize
LIB_DIR = 'D:\Hannes Horneber\Documents\Wissen\Studium\Computer Science\WS17-18 STD\code\projects\win_tf_unet\eval\functions';

% addpath functions
% addpath functions/helpers
addpath (genpath(LIB_DIR));
% addpath functions/metrics
totalTime = tic;

%% Settings I

% dir with net output: tile, groundtruth, label, softmax, (uncertainty)
% NET_OUTPUT = 'data_net/ex4_trainset';
% NET_OUTPUT = 'data_net/ex5b_testset';
% NET_OUTPUT = 'data_net/ex4_trainset';
% NET_OUTPUT = 'data_all\unet_23_0227_bn9_sADp020_data1024\pred_04-23_0815_bn9_Dp020'
% NET_OUTPUT = 'data_all\unet_23_0227_bn9_sADp020_data1024\pred_04-23_1601_bn9_Dp020_30000'
% NET_OUTPUT = 'data_all\unet_15_2028_bn9_sA\pred_04-25_0701_bn9__5000';
% NET_OUTPUT = 'data_all\unet_15_2028_bn9_sA\pred_04-25_0704_bn9__30000';
% NET_OUTPUT = 'data_tested/unet_15_2030_bn9_sADp050/pred_04-25_0750_bn9_Dp050_5000'
NET_OUTPUT = 'D:\Hannes Horneber\Documents\Wissen\STUDIUM\Computer Science\WS17-18 STD\code\projects\EVAL\data_test\unet_21_1337_bn9_sADp050_AL50\pred_04-25_1922_bn9__AL50_30000';
%% Settings I
% specify directory to write outputs of script
DIR_RES = [ NET_OUTPUT '/rerun' datestr(now,'yyyy-mm-dd-HH-MM-SS') ]; 

% ##############################>   OUTPUT OPTIONS    <####################
% OUTPUT OPTIONS (choose output: toggle true/false to plot or not)
OUT_TILE =          true;      % tile itself
OUT_UNC =           true;       % uncertainty output
OUT_SOFT =          true;       % softmax output
OUT_SOFT_ENTR =     false;       % softmax entropy
OUT_OVL =           true;      % tile with segmentation overlay
OUT_SEG =           true;      % segmentation itself
OUT_TRUTH =         true;      % [labeled data] groundtruth
OUT_IOU =           true;      % [labeled data] IoU map
OUT_OBJ =           true;      % [labeled data] object based stats
OUT_OBJ_PLOT =      false;      % [labeled data] plot of iou_map with obj
OUT_PIX_ROC =       true;      % [labeled data] pixelwise ROC
OUT_PIX_BIN =       false;      % calibration plot
OUT_FIG_CONF =      true;      % [labeled data] figure with confusion matrix
OUT_FIG_CONF_PI =   false;      % [labeled data] ... per image
OUT_FIG_ROC =       false;      % [labeled data] figure with ROC
OUT_OUTPUT_LAYER =  false;      % pseudo-probability map for each class

OBJ_IOU_THRESH =    0.1;        % for obj based (for which IoU is an obj "found")
N_BINS =            10;

N_QUANTILES =       100;
MAX_SCORE =         1.0;        % score max for scaling quantiles
REMOVE_LOW_AP =     0.1;        % for removing bad curves
INTERP_SAMPLES =    120;        % for averaging over curves
INTERP_SAMPLES_PLOT = 360;      % for plotting averaged curve
interp_x =    0:0.007:1;         % 143 samples
interp_smooth_x = 0:0.03:1;      % 34 samples

% specify filetype in which output graphics are generated
% '-djpeg' / '-dtiff' / if saving with: print(handle, filename, format)
% '.png' / '-jpg' / '.tiff' / if saving with: imwrite(data, filename)
OUT_TILE_FORMAT =	'.png';     % use .tif for quality, .jpg for small size
OUT_SEG_FORMAT =	'.png';    % use .tiff for quality, .png for small size
OUT_FIG_FORMAT =	'.png';     % use .png! .jpg is crap for graphics

% number of input/output classes (for stats)
NCLASSES =      3;  % 3 = bg, tree, ignore
NOUTCLASSES =	2;  % 2 = bg, tree (usually NCLASSES - ignore)
% segmentation colormap
CMAP_SEG = [0,0,0;165,215,0;255,255,255]/255; % bg, dw_standing, ignore
CMAP_OVL_TRUTH = [0,0,0;0,215,165;255,255,255]/255; % bg, dw_standing, ignore
CMAP_OVL_PRED = [0,0,0;165,215,0;255,255,255]/255; % bg, dw_standing, ignore

% determine whether (raw) data needs to be normalized (0-255 => 0-1)
NORMALIZE = true; % usually true for int and false for float input (f32/f64)

VERBOSE = true;

%%
fprintf('Load py arrays from %s\n', NET_OUTPUT);
tmp = dir([NET_OUTPUT '/*.mat']);
data = {tmp.name}';
% expand file name to full path
data = cellfun(@(file)[NET_OUTPUT '/' file],data,'uni',false);

data_n = length(data);
clear tmp

%% Generate Metrics / Graphs
%##########################################################################
% create results folder
mkdir(DIR_RES);
% determine number of leading zeros from length of img_idx for filenames
TFORMAT = ['%0' num2str(numel(num2str(length(data_n)))) '.f'];

% allocate for stats
% iou = zeros(length(img_idx),NOUTCLASSES);
% iou_stats = zeros(length(img_idx),(NOUTCLASSES*4 + NOUTCLASSES-1)); % TP, FP, FN, TN, (FC)
% elapsed_time = zeros(length(img_idx), 1);
row = 1; % rownr for indexing arrays
% n graphs * n quantiles (row) * (3 cols: precision, recall)

%bin_scores = (0:N_BINS-1) * MAX_SCORE / N_BINS;
% 1 = recall, 2 = precision, 
% 3 = overall accuracy with increasing uncertainty (tp, tn / fp, fn)
% 4 = frequency of correct classifications per bin (c1)
% 5 = frequency of correct classifications per bin (both)
graphs_uncertainty = zeros([data_n, N_QUANTILES, 5]);
graphs_softmax = zeros([data_n, N_QUANTILES, 5]);
quantiles = 0:N_QUANTILES-1;
quantiles = quantiles * MAX_SCORE / N_QUANTILES;


%%
% iterate through data: img_i = 1:length(img_idx) -> img_idx(img_i)
for tile_i = 1:data_n
    % ##########################> LOAD DATA <##############################
    [~, tile_name, ~] = fileparts(data{tile_i});
    tile_name = [DIR_RES '/' tile_name];
    
    %tile, label, pred, softmax 
    load(data{tile_i})
    if(VERBOSE); fprintf('Loaded %s\n', tile_name); end
    tile = double(tile);
    label = boolean(label); %uint8(label);
    softmax = double(softmax);
    pred = boolean(pred); %uint8(pred);
    % double channel so it has same shape as softmax (makes using it as score easier)
    uncertainty(:,:,2) = double(uncertainty(:,:));
    

    % #########################> PRINT DATA  <#############################
    % print original tile (RGB and/or IRGB)
    if(OUT_TILE)
        % write tile without georef
        imwrite(tile(:,:,[1 2 3]), [tile_name '-a' OUT_TILE_FORMAT]);
    end
    if(OUT_UNC)
        imwrite(imagesc2im(uncertainty(:,:,1), 0), [tile_name '-u' OUT_TILE_FORMAT]);
    end
    if(OUT_SOFT)
        imwrite(imagesc2im(softmax(:,:,2), 0), [tile_name '-sm' OUT_TILE_FORMAT]);
    end
    if(OUT_SOFT_ENTR)
        sm_entropy = - (softmax(:,:,1) * log(softmax(:,:,1)) +  (softmax(:,:,2) * log(softmax(:,:,2))));
        sm_entropy = sm_entropy / max(sm_entropy(:));
        sm_entropy(:,:,2) = sm_entropy;
        imwrite(imagesc2im(sm_entropy(:,:,1), 0), [tile_name '-sme' OUT_TILE_FORMAT]);
    end

    % ##########################> PROCESS PREDICTION <#####################
    % work with network output: create files, stats and graphs
    
    % write img: segmentation (indexed color file)
    if(OUT_SEG)
        % subtract 1 so that labels are 0:3 / 0:4 for output files
        % (consistent with groundtruth labels)
        if(VERBOSE); fprintf('  > predicted segmentation\n'); end
        imwrite(pred, CMAP_SEG, [tile_name '-seg' OUT_SEG_FORMAT]);
    end
    
    % write img: segmentation overlay
    if(OUT_OVL)
        if(VERBOSE); fprintf('  > overlay tile with prediction\n'); end
        ovl = tile;
        % output labels are 1:3, ground truth labels are 0:2 (-> add 1)
        for desired_class = 1 % not for bg (start at 2), only for std
            cmap_ovl = CMAP_OVL_TRUTH;
            % create overlay mask groundtruth labels
            mask = label == desired_class;
            mask = mask - imerode(mask, ones(7,7));
            mask = logical(mask);

            % slice mask pixels in each color channel
            for channel = 1:3 % RGB
                slice = ovl(:,:,channel);
                slice(mask) = cmap_ovl(desired_class+1,channel);
                ovl(:,:,channel) = slice;
            end; clear channel slice;
            
            cmap_ovl = CMAP_OVL_PRED;
            % create overlay mask
            mask = pred == desired_class;
            mask = mask - imerode(mask, ones(7,7));
            mask = logical(mask);

            % slice mask pixels in each color channel
            for channel = 1:3 % RGB
                slice = ovl(:,:,channel);
                slice(mask) = cmap_ovl(desired_class+1,channel);
                ovl(:,:,channel) = slice;
            end; clear channel slice;
        end; clear desired_class;
        
        % save tile with overlay (use RGB: [1 2 3] or RGBI: [4 1 3])      
        imwrite(ovl(:,:,[1 2 3]), [tile_name '-ovl' OUT_TILE_FORMAT]);
    end; clear ovl;

    % ##########################> PROCESS GROUNDTRUTH <####################
    % comparison of groundtruth and segmentation prediction (for IoU, ...)

    % write img: groundtruth labels
    if(OUT_TRUTH)
        if(VERBOSE); fprintf('  > groundtruth / labelmask\n'); end
    	imwrite(label, CMAP_SEG, [tile_name '-truth' OUT_SEG_FORMAT]);
    end

    % create iou image
    if(OUT_IOU || OUT_OBJ_PLOT)
        iou_map = zeros([size(label) 3]);

        for desired_class = 0:NOUTCLASSES-1 % for all classes
            % output labels are 1:3, ground truth labels are 0:2 (-> add 1)
            [iou(row,desired_class+1), iou_map_c, stats_c, stats_fpc] = iouIMGPerClass( label, pred, desired_class );

            % insert additional iou_stats to global array
            iou_stats(row,1+(desired_class)*4:(desired_class+1)*4) = stats_c;
            if(desired_class ~= 0)
                 % for non bg-class add further iou_stats to global array
                iou_stats(row,(NOUTCLASSES*4 + desired_class+1)) = stats_fpc;
            end

            % add iou_map for class to global iou_map
            iou_map = iou_map + iou_map_c;
            if(VERBOSE); fprintf('  > pixel IoU tile %i, class %i: %f\n', tile_i, desired_class, iou(row,desired_class+1)); end  
        end; clear desired_class;

        % print IoU map
        if(OUT_IOU)
            imwrite(iou_map, [tile_name '-t_iou' OUT_TILE_FORMAT]);
        end
    end; clear iou_map_c;
    
    % #########################> PROCESS OBJECT_BASED <####################
    % comparison of groundtruth and segmentation prediction (for IoU, ...)

    % OBJECT BASED
    if(OUT_OBJ)       
        if(VERBOSE); fprintf('  > object based\n'); end
        class = 1;
        % binary images of groundtruth/pred
        groundtruth_classmask = false(size(label));
        groundtruth_classmask(label == class) = true;

        prediction_classmask = false(size(pred));
        prediction_classmask(pred == class) = true;

        % create objects
        [objPred, objPred_map] = im2obj(prediction_classmask); 
        %visObjects( prediction_classmask, objPred, 'id' );
        [objTruth, objTruth_map] = im2obj(groundtruth_classmask, 'MODE', 'watershed'); 
        %visObjects( groundtruth_classmask, objTruth, 'id' );

        if(VERBOSE); fprintf('   >> Class %i: Found %i/%i objects.\n', class, length(objPred), length(objTruth)); end  

        % compute IoU for predicted objects
        if(length(objPred) > 0)
            objPred = addObjMetrics(objPred, objTruth);
            
            % only plot if obj are present
            if(OUT_OBJ_PLOT)
                % objplot = plotObjects(imbinarize(label), objPred, '@composite');
                % objplot = plotObjects(softmax_map, objPred, '@composite');
                objplot = plotObjects(iou_map, objPred, '@composite');
                saveas(objplot, [tile_name '_obj_pred' OUT_FIG_FORMAT]);
                close(objplot)
            end
        end

        softmax_map = softmax(:,:,class+1); % softmax_map for class
        %softmax_map_bg = softmax_netoutput(:,:,1); % softmax_map for background
        %softmax_img = imagesc2im(softmax_map, summer(256));
        uncertainty_map = uncertainty(:,:,class+1);
        
        % assign scores to objects
        for i=1:length(objPred)   
            objPred(i).score = mean(softmax_map(objPred_map == i));
            objPred(i).uncertainty = mean(uncertainty_map(objPred_map == i));
        end


        if ~exist('objPred_all','var') || length(objPred_all) < class
            % initialize object container
            objPred_all{class} = objPred;
            objTruth_all{class} = objTruth;
        else
            % offset IDs to append to struct
            offsetPred = length(objPred_all{class});
            offsetTruth = length(objTruth_all{class});
            for i = 1:length(objPred) % doesn't work: deal([objPred.id] + offsetPred);
                    objPred(i).id = objPred(i).id + offsetPred;
                    objPred(i).overlaps = objPred(i).overlaps + offsetTruth;
            end
            for i = 1:length(objTruth) % doesn't work: deal([objPred.id] + offsetPred);
                    objTruth(i).id = objTruth(i).id + offsetTruth;
            end

            % append to global obj store
            if ~length(objPred) == 0
                if length(objPred_all{class}) > 0
                    objPred_all{class} = [objPred_all{class}; objPred];
                else
                    objPred_all{class} = objPred;
                end
            end
            if ~length(objTruth) == 0
                if length(objTruth_all{class}) > 0
                    objTruth_all{class} = [objTruth_all{class}; objTruth];
                else
                    objTruth_all{class} = objTruth;
                end
            end
        end
    end; 
    clear class objPred objTruth objPred_map objTruth_map;
    clear softmax_map softmax_netoutput;

    % ###########################> PROCESS PIXELWISE <#####################
    % comparison of groundtruth and segmentation prediction (for IoU, ...)
    
    % PIXELWISE PRECISION/RECALL
    if(OUT_PIX_ROC)
%         figure(); imshow(label);
%         figure(); imshow(pred);
%         figure(); imagesc(uncertainty(:,:,1))

        for g = 1:2
            if g==1; 
                score = uncertainty(:,:,1); 
            else
                score = 1 - softmax(:,:,2);
%               score = sm_entropy(:,:,1);
            end
            % allocate graph
            %graph(q_steps) = struct('quantile',[],'precision',[],'recall',[]);
            if nnz(label) && nnz(pred)
%                  myplot = figure();
%                  hold on;
                % fill graph
                for q = 1:N_QUANTILES
                    %graphs_uncertainty(tile_i,q,) = max_score * (q/q_steps);
    %                 fprintf('showing quantile %.2f\n', quantiles(q));
                    % select pixels of quantile:
                    % -> remove all pixels that have higher uncertainty
                    label_q = label;
                    label_q(score > quantiles(q)) = false;
                    pred_q = pred;
                    pred_q(score > quantiles(q)) = false;
                    %imshow(label_q); waitforbuttonpress;

                    if nnz(label_q) && nnz(pred_q)
                        % true/false positives (precision) and false negatives (recall)
                        tp = label_q;
                        tp(~pred_q) = false;

                        fn = label; % calc recall on all pixels, not just quantile
                        fn(pred_q) = false;

                        fp = pred_q;
                        fp(label_q) = false;
                    
                        precision_val = nnz(tp) / (nnz(tp) + nnz(fp));
                        recall_val = nnz(tp) / (nnz(tp) + nnz(fn));
                    elseif ~nnz(pred_q)
                        precision_val = 1.0;
                        recall_val = 0.0;
                    elseif ~nnz(label_q) && nnz(pred_q)
                        precision_val = 0.0;
                        recall_val = 0.0;
                    end

                    % compute prec/recall
                    if g==1; graphs_uncertainty(tile_i,q,1) = precision_val;
                    else graphs_softmax(tile_i,q,1) = precision_val; end

                    if g==1; graphs_uncertainty(tile_i,q,2) = recall_val;
                    else graphs_softmax(tile_i,q,2) = recall_val; end
    %                 fprintf('precision %.3f / recall %.3f\n', precision_val, recall_val);
                    
                    % -------------------------------------------------
                    % overall accuracy with increasing uncertainty
                    preds = pred(score <= quantiles(q));
                    truth = label(score <= quantiles(q));
                    pred_tp_fn = preds(truth == true);
                    truth_tp_fn = truth(truth == true);
                    pred_tp_fp = preds(preds == true);
                    truth_tp_fp = truth(preds == true);
                    
                    correct_tp_fn = nnz(pred_tp_fn == truth_tp_fn);
%                     accuracy_tp_fn = correct_tp_fn / length(pred_tp_fn);
                    
                    correct_tp_fp = nnz(pred_tp_fp == truth_tp_fp);
%                     accuracy_tp_fp = correct_tp_fp / length(pred_tp_fp);
                    
                    correct_tp_fp = nnz(pred_tp_fp == truth_tp_fp) + nnz(pred_tp_fn == truth_tp_fn);
                    accuracy = correct_tp_fp / (length(truth_tp_fp) + length(truth_tp_fn));
                    
                    if g==1; graphs_uncertainty(tile_i,q,3) = accuracy;
                    else graphs_softmax(tile_i,q,3) = accuracy; end
                    
                    % accuracy per uncertainty bin
                    if q == 1
                        bin_start = quantiles(q);
                    else
                        bin_start = quantiles(q - 1);
                    end
                    bin_end = quantiles(q);
                    
                    score_mask = true(size(score));
                    score_mask(score <= bin_start) = false;
                    score_mask(score > bin_end) = false;
    
                    preds = pred(score_mask);
                    truth = label(score_mask);
                    pred_tp_fn = preds(truth == true);
                    truth_tp_fn = truth(truth == true);
                    pred_tp_fp = preds(preds == true);
                    truth_tp_fp = truth(preds == true);
                    
                    correct_tp_fn = nnz(pred_tp_fn == truth_tp_fn);
                    accuracy_tp_fn = correct_tp_fn / length(pred_tp_fn);
                    
                    correct_tp_fp = nnz(pred_tp_fp == truth_tp_fp);
                    accuracy_tp_fp = correct_tp_fp / length(pred_tp_fp);
                    
                    correct_tp_fp = nnz(pred_tp_fp == truth_tp_fp) + nnz(pred_tp_fn == truth_tp_fn);
                    bin_accuracy = correct_tp_fp / (length(truth_tp_fp) + length(truth_tp_fn));
                    
%                     label_bin = label;
%                     label_bin(~score_mask) = false;
%                     pred_bin = pred;
%                     pred_bin(~score_mask) = false;        
%                     mask = zeros([size(label) 3]);
%                     red = zeros(size(label));
%                     red(score_mask) = true;
%                     mask(:,:,1) = red;
%                     mask(:,:,2) = label_bin;
%                     mask(:,:,3) = pred_bin;
%                     imshow(mask); 
%                     title(['show | fn: ' num2str(accuracy_tp_fn) ' | fp: ' num2str(accuracy_tp_fp) ' | all: ' num2str(bin_accuracy)]);
%                     waitforbuttonpress;
                                        
                    if g==1; graphs_uncertainty(tile_i,q,4) = bin_accuracy;
                    else graphs_softmax(tile_i,q,4) = bin_accuracy; end
                    
                    
                    % accuracy per uncertainty bin
                    if q == 1
                        bin_start = quantiles(q);
                    else
                        bin_start = quantiles(q - 1);
                    end
                    bin_end = quantiles(q);
                    
                    score_mask = true(size(score));
                    score_mask(score <= bin_start) = false;
                    score_mask(score > bin_end) = false;
    
                    preds = pred(score_mask);
                    truth = label(score_mask);
                    
                    correct = nnz(preds == truth);
                    bin_accuracy2 = correct / length(truth);
                         
%                     label_bin = label;
%                     label_bin(~score_mask) = false;
%                     pred_bin = pred;
%                     pred_bin(~score_mask) = false;        
%                     mask = zeros([size(label) 3]);
%                     red = zeros(size(label));
%                     red(score_mask) = true;
%                     mask(:,:,1) = red;
%                     mask(:,:,2) = label_bin;
%                     mask(:,:,3) = pred_bin;
%                     imshow(mask); 
%                     title(['show | pixels: ' num2str( length(truth)) ' | correct: ' num2str(correct) ' | all: ' num2str(bin_accuracy2)]);
%                     waitforbuttonpress;
                                        
                    if g==1; graphs_uncertainty(tile_i,q,5) = bin_accuracy2;
                    else graphs_softmax(tile_i,q,5) = bin_accuracy2; end
                    
                end; clear precision_val recall_val
            else
                fprintf('    skipping OUT_PIX_ROC\n');
            end
        end; clear g
        
%         plot(quantiles,graphs_softmax(tile_i,:,4))
%         plot(quantiles,graphs_softmax(tile_i,:,5))
%         figure(); imshow(label);
%         figure(); imshow(pred);
    end
    
    
    % CONFUSION MATRIX
    if(OUT_FIG_CONF || OUT_FIG_ROC)
        % convert to oneHot encoding (&flatten)
        groundtruth_oneHot = oneHot(label(:), 0:NOUTCLASSES-1);
        segmentation_oneHot = oneHot(pred(:), 0:NOUTCLASSES-1);

        % collect results for global confusion matrix over the whole dataset
        if(exist('summary_groundtruth_oneHot', 'var'))
            summary_groundtruth_oneHot = [summary_groundtruth_oneHot;groundtruth_oneHot];
            summary_segmentation_oneHot = [summary_segmentation_oneHot;segmentation_oneHot];
        else
            summary_groundtruth_oneHot = groundtruth_oneHot;
            summary_segmentation_oneHot = segmentation_oneHot;
        end

        % create confusion matrix plot for this tile
        % transpose inputs! (input NxM matrix with N classes, M samples)
        if(OUT_FIG_CONF_PI)
            confusionplot = plotconfusion(groundtruth_oneHot',segmentation_oneHot');
            saveas(confusionplot, [tile_name '_cp' OUT_FIG_FORMAT]);
        end; clear confusionplot;

        if(OUT_FIG_ROC)
            rocplot = plotroc(groundtruth_oneHot',segmentation_oneHot');
            saveas(rocplot, [tile_name '_rp' OUT_FIG_FORMAT]);
        end; clear rocplot;
    end; clear groundtruth_oneHot segmentation_oneHot;

    % increase rownr for indexing iou array
    row = row + 1;
end % end loading and evaluating single tiles
% #########################################################################

%% global pixelwise ROC out
if(OUT_PIX_ROC)
    clear prec_vals recall_vals bin_acc_vals
    % ###############################################
    % UNCERTAINTY GRAPH
    curves = 0;
    for tile_i = 1:data_n
        tmp_prec_vals = graphs_uncertainty(tile_i,:,1);
        tmp_recall_vals = graphs_uncertainty(tile_i,:,2);

        % discard NaN and all-zero rows        
        if ~any(isnan(graphs_uncertainty(tile_i,:,1))) && any(graphs_uncertainty(tile_i,:,1)~=0) && any(graphs_uncertainty(tile_i,:,1)~=1)
            AP = trapz(graphs_uncertainty(tile_i,:,2),  graphs_uncertainty(tile_i,:,1));
            fprintf(['tile ' num2str(tile_i) ' | AP: ' num2str(AP) '\n']);
            
            % handle curves that are flat
            if AP == 0
                [recall_vals_unique, ifirst, ~] = unique(tmp_recall_vals);
                if length(recall_vals_unique) == 1
                    tmp_recall_vals(ifirst) = 0.0;
                    tmp_recall_vals(length(tmp_recall_vals)) = recall_vals_unique(1) + 0.0005;
                    tmp_prec_vals(length(tmp_prec_vals)) = 0.0;
                    AP = trapz(tmp_recall_vals,  tmp_prec_vals);
                    fprintf(['fixed tile ' num2str(tile_i) ' | AP: ' num2str(AP) '\n']);
                end
            end
            
            if AP > REMOVE_LOW_AP
                curves = curves + 1;           
                prec_vals(curves, :) = tmp_prec_vals; %graphs_uncertainty(tile_i,:,1);
                recall_vals(curves, :) = tmp_recall_vals; %graphs_uncertainty(tile_i,:,2);
                % for calib plot
                bin_acc_vals(curves, :) = graphs_uncertainty(tile_i,:,5);
                
                recall_max = max(recall_vals(curves, :));
                [recall_vals_unique, ia, ~] = unique(recall_vals(curves, :));
                % interpolate if valid curve
                if length(recall_vals_unique) >= 3
                    prec_vals_unique = prec_vals(curves, ia);
                    % add zero entries to guide interpolation
                    recall_vals_unique(length(recall_vals_unique)+1) = recall_max + 0.001;
                    prec_vals_unique(length(recall_vals_unique+1)) = 0.0;
                    recall_vals_unique(length(recall_vals_unique)+1) = 1.0;
                    prec_vals_unique(length(recall_vals_unique+1)) = 0.0;
                    prec_vals_interp(curves, :) = interp1(recall_vals_unique, prec_vals_unique, interp_x, 'previous', 'extrap');
                end
            end; clear ia recall_max recall_vals_unique prec_vals_unique
        end
    end
    % remove bad interpolations
    prec_vals_interp_clean = [];
    for c = 1:curves
        interpolated = prec_vals_interp(c, :);
        if nnz(interpolated)
            interpolated(isnan(interpolated)) = 1.0;
            prec_vals_interp_clean(size(prec_vals_interp_clean, 1)+1, :) = interpolated;
        end
    end; clear c interpolated
    
    % -------------------------
    rocplot = figure();
    hold on;
    % plot ROC imagewise
    for tile_i = 1:curves
        plot(recall_vals(tile_i, :), prec_vals(tile_i, :), 'Color', [0.7, 0.7, 0.7]); % actual graph
    end
    % plot for average interpolated lines
    % for tile_i = 1:size(prec_vals_interp_clean, 1)
    %     plot(recall_interp, prec_vals_interp_clean(tile_i, :), 'Color', [0.3, 0.3, 0.7]); % actual graph
    % end
    % plot average
%     AP_unc = trapz(interp_x, mean(prec_vals_interp_clean, 1)); % calc area under curve
%     plot(interp_x, mean(prec_vals_interp_clean, 1), 'LineWidth', 1, 'Color', [0.1 0.2 0.7]); % actual graph
    % plot smooth average
    prec_vals_interp_smooth = interp1(interp_x, mean(prec_vals_interp_clean, 1), interp_smooth_x, 'spline');
    AP_unc2 = trapz(interp_smooth_x, prec_vals_interp_smooth); % calc area under curve
    plot(interp_smooth_x, prec_vals_interp_smooth, 'LineWidth', 2, 'Color', [0.1 0.2 0.7]); % actual graph
    line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
    xlabel('recall') % x-axis label
    ylabel('precision') % y-axis label
    title(['Precision/Recall with increasing uncertainty | AP: ' num2str(AP_unc2)]);
%     axis([0.0, max(recall_vals(:))+0.1, 0.0, 1.0]);
    axis([0.0, 1.0, 0.0, 1.0]);
    grid on;
    
    outname = [DIR_RES '/' '_ROC_pix_uncertainty_ctrl' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
    print( '-dpng', '-r200', outname);
    close(rocplot);
    % ------------------------- ROC
    rocplot = figure();
    hold on;
    % plot smooth average
    prec_vals_interp_smooth = interp1(interp_x, mean(prec_vals_interp_clean, 1), interp_smooth_x, 'spline');
    AP_unc2 = trapz(interp_smooth_x, prec_vals_interp_smooth); % calc area under curve
    plot(interp_smooth_x, prec_vals_interp_smooth, 'LineWidth', 2, 'Color', [0.1 0.2 0.7]); % actual graph
    line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
    xlabel('recall') % x-axis label
    ylabel('precision') % y-axis label
    title(['Precision/Recall with increasing uncertainty | AP: ' num2str(AP_unc2)]);
%     axis([0.0, max(recall_vals(:))+0.1, 0.0, 1.0]);
    axis([0.0, 1.0, 0.0, 1.0]);
    grid on;
    
    outname = [DIR_RES '/' '_ROC_pix_uncertainty' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
    print( '-dpng', '-r200', outname);
    close(rocplot);
    % -------------------------
    if(OUT_PIX_BIN)
        % ------------------------- BIN CALIB
        binplot = figure();
        hold on;
        for tile_i = 1:curves
            plot(quantiles, bin_acc_vals(curves, :), 'Color', [0.7, 0.7, 0.7]); % per image plot
        end
        % plot smooth average
        plot(interp_smooth_x, mean(bin_acc_vals, 1), 'LineWidth', 2, 'Color', [0.1 0.2 0.7]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('certainty') % x-axis label
        ylabel('frequency') % y-axis label
        title(['uncertainty calibration']);
        axis([0.0, 1.0, 0.0, 1.0]);
        grid on;

        outname = [NET_OUTPUT '/' '_CAL_pix_uncertainty' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(binplot);
        % -------------------------
    end
    
    clear prec_vals2 recall_vals2 prec_vals_interp2
    % ###############################################
    % SOFTMAX GRAPH
    curves2 = 0;
    for tile_i = 1:data_n
        % discard NaN and all-zero rows
        if ~any(isnan(graphs_softmax(tile_i,:,1))) && any(graphs_softmax(tile_i,:,1)~=0) && any(graphs_softmax(tile_i,:,1)~=1)
            AP = trapz(graphs_softmax(tile_i,:,2),  graphs_softmax(tile_i,:,1));
            fprintf(['softmax tile ' num2str(tile_i) ' | AP: ' num2str(AP) '\n']);
            %plot(graphs_softmax(tile_i,:,2),  graphs_softmax(tile_i,:,1));
            

            if AP > REMOVE_LOW_AP
                curves2 = curves2 + 1;
                prec_vals2(curves2, :) = graphs_softmax(tile_i,:,1);
                recall_vals2(curves2, :) = graphs_softmax(tile_i,:,2);

                AP = trapz( recall_vals2(curves2, :),  prec_vals2(curves2, :));
                fprintf(['tile ' num2str(tile_i) ' | AP: ' num2str(AP) '\n']);

                recall_max = max(recall_vals2(curves2, :));
                [recall_vals_unique, ia, ~] = unique(recall_vals2(curves2, :));
                % interpolate if valid curve
                if length(recall_vals_unique) > 5
                    prec_vals_unique = prec_vals2(curves2, ia);
                    % add zero entries to guide interpolation
                    recall_vals_unique(length(recall_vals_unique)+1) = recall_max + 0.001;
                    prec_vals_unique(length(recall_vals_unique+1)) = 0.0;
                    recall_vals_unique(length(recall_vals_unique)+1) = 1.0;
                    prec_vals_unique(length(recall_vals_unique+1)) = 0.0;
                    prec_vals_interp2(curves2, :) = interp1(recall_vals_unique, prec_vals_unique, interp_x, 'previous', 'extrap');
                end
            end; clear recall_max recall_vals_unique prec_vals_unique
        end
    end
    % remove bad interpolations
    prec_vals_interp_clean2 = [];
    for c = 1:curves2
        interpolated = prec_vals_interp2(c, :);
        
        if nnz(interpolated)
            interpolated(isnan(interpolated)) = 1.0;
            prec_vals_interp_clean2(size(prec_vals_interp_clean2, 1)+1, :) = interpolated;
        end
    end; clear c interpolated
    
    if curves2
        % -------------------------
        rocplot2 = figure();
        hold on;
        % plot ROC imagewise
        for tile_i = 1:curves2
            plot(recall_vals2(tile_i, :), prec_vals2(tile_i, :), 'Color', [0.7, 0.7, 0.2]); % actual graph
        end
        % plot for average interpolated lines
        for tile_i = 1:size(prec_vals_interp_clean2, 1)
            plot(interp_x, prec_vals_interp_clean2(tile_i, :), 'Color', [0.7 0.7 0.2]); % actual graph
        end
        % plot average
%         AP_soft = trapz(interp_x, mean(prec_vals_interp_clean2, 1)); % calc area under curve
%         plot(interp_x, mean(prec_vals_interp_clean2, 1), 'LineWidth', 1, 'Color', [0.7 0.1 0.2]); % actual graph
        % smooth average
        prec_vals_interp_smooth2 = interp1(interp_x, mean(prec_vals_interp_clean2, 1), interp_smooth_x, 'spline');
        AP_soft2 = trapz(interp_smooth_x, prec_vals_interp_smooth2); % calc area under curve
        plot(interp_smooth_x, prec_vals_interp_smooth2, 'LineWidth', 2, 'Color', [0.7 0.1 0.2]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('recall') % x-axis label
        ylabel('precision') % y-axis label
        title(['ROC for softmax | AP: ' num2str(AP_soft2)]);
%         axis([0.0, max(recall_vals2(:))+0.1, 0.0, 1.0]);
        axis([0.0, 1.0, 0.0, 1.0]);
        grid on;

        outname = [DIR_RES '/' '_ROC_pix_softmax_ctrl' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot2);
        
        % -------------------------
        rocplot3 = figure();
        hold on;
        % smooth average
        prec_vals_interp_smooth2 = interp1(interp_x, mean(prec_vals_interp_clean2, 1), interp_smooth_x, 'spline');
        AP_soft2 = trapz(interp_smooth_x, prec_vals_interp_smooth2); % calc area under curve
        plot(interp_smooth_x, prec_vals_interp_smooth2, 'LineWidth', 2, 'Color', [0.7 0.1 0.2]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('recall') % x-axis label
        ylabel('precision') % y-axis label
        title(['Precision/Recall for decreasing softmax | AP: ' num2str(AP_soft2)]);
%         axis([0.0, max(recall_vals2(:))+0.1, 0.0, 1.0]);
        axis([0.0, 1.0, 0.0, 1.0]);
        grid on;

        outname = [DIR_RES '/' '_ROC_pix_softmax' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot3);
        % -------------------------
        % ###############################################
        % COMBINED GRAPH
        % -------------------------
        rocplot3 = figure();
        hold on;
        % plot ROC imagewise
        % for tile_i = 1:curves
        %     plot(recall_vals(tile_i, :), prec_vals(tile_i, :), 'Color', [0.8, 0.8, 0.8]); % actual graph
        % end
        % plot average of uncertainty ROCs
        %plot(interp_x, mean(prec_vals_interp_clean2, 1), 'LineWidth', 2); % actual graph
        plot(interp_smooth_x, prec_vals_interp_smooth, 'LineWidth', 2, 'Color', [0.1 0.2 0.7]); % actual graph
        % plot average of softmax ROCs
        %plot(interp_x, mean(prec_vals_interp_clean, 1), 'LineWidth', 2); % actual graph
        plot(interp_smooth_x, prec_vals_interp_smooth2, 'LineWidth', 2, 'Color', [0.7 0.1 0.2]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('recall') % x-axis label
        ylabel('precision') % y-axis label
        title(['PR for Uncertainty/Softmax | APunc ' num2str(AP_unc2) ' | APsoft '  num2str(AP_soft2)]);
        axis([0.0, 1.0, 0.0, 1.0]);
        grid on;

        outname = [NET_OUTPUT '/' '_ROC_pix_both' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot3);
        % -------------------------
        
        % ###############################################
        % DIFFERENCE GRAPH
       
        % ------------------------
        %diff_val = mean(prec_vals_interp_clean, 1) - mean(prec_vals_interp_clean2, 1);
        diff_val_smooth = prec_vals_interp_smooth - prec_vals_interp_smooth2;
        AP_diff = trapz(interp_smooth_x, diff_val_smooth);
        diffplot = figure();
        hold on;
        %plot(interp_x, diff_val, 'LineWidth', 2); % actual graph
        plot(interp_smooth_x, diff_val_smooth, 'LineWidth', 2, 'Color', [0.6 0.2 0.6]); % actual graph
        xlabel('recall') % x-axis label
        ylabel('diff') % y-axis label
        title(['Expressiveness: Softmax vs. uncertainty  | AP ' num2str(AP_diff)]);
%         axis([0.0, 1.0, 0.0, 1.0]);
        grid on;

        outname = [NET_OUTPUT '/' '_ROC_pix_diff' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(diffplot);
        % -------------------------
        
        % -------------------------
        % SAVE VARIABLES
        outname = [DIR_RES '/' '_pix_graph' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS')];
        save(outname, 'interp_smooth_x', 'prec_vals_interp_smooth', 'prec_vals_interp_smooth2', 'diff_val_smooth', 'NET_OUTPUT')
        % -------------------------
    end
end

%% global obj out

if(OUT_OBJ)
    for class = 1
        objPredClass = objPred_all{class};
        objTruthClass = objTruth_all{class};
        % allow sorted access, start with highest score
        [~,ord] = sort([objPredClass.score], 'descend');
        objPred_sorted = objPredClass(ord);

        for i=1:length(objPred_sorted)
            objPred_sorted(i).recall = length(find(vertcat(objPred_sorted(1:i).iou) > OBJ_IOU_THRESH)) / ...
                length(objTruthClass);
            objPred_sorted(i).precision = length(find(vertcat(objPred_sorted(1:i).iou) > OBJ_IOU_THRESH)) / i;
        end; clear i

        save('obj.mat','objPred_all','objTruth_all')

        % calc area under curve
        AP = trapz(vertcat(objPred_sorted.recall), vertcat(objPred_sorted.precision));
        MR = max(vertcat(objPred_sorted.recall));
        % -------------------------
        rocplot = figure();
        hold on;
        plot(vertcat(objPred_sorted.recall), vertcat(objPred_sorted.precision), 'Color', [0.7 0.1 0.2]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('recall') % x-axis label
        ylabel('precision') % y-axis label
        title(['Precision/Recall with decreasing softmax | AP ' num2str(AP) ' | MR ' num2str(MR)]);
        axis([0, 1, 0, 1]);
        grid on;
        
        outname = [NET_OUTPUT '/' '_ROC_obj_softmax' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot);
        % -------------------------
        
        % repeat for uncertainty
        [~,ord] = sort([objPredClass.uncertainty], 'ascend');
        objPred_sorted2 = objPredClass(ord);

        for i=1:length(objPred_sorted2)
            objPred_sorted2(i).recall = length(find(vertcat(objPred_sorted2(1:i).iou) > OBJ_IOU_THRESH)) / ...
                length(objTruthClass);
            objPred_sorted2(i).precision = length(find(vertcat(objPred_sorted2(1:i).iou) > OBJ_IOU_THRESH)) / i;
        end; clear i

        save('obj.mat','objPred_all','objTruth_all')

        % calc area under curve
        AP2 = trapz(vertcat(objPred_sorted2.recall), vertcat(objPred_sorted2.precision));
        MR2 = max(vertcat(objPred_sorted.recall));
        % -------------------------
        rocplot = figure();
        hold on;
        plot(vertcat(objPred_sorted2.recall), vertcat(objPred_sorted2.precision), 'Color', [0.1 0.2 0.7]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('recall') % x-axis label
        ylabel('precision') % y-axis label
        title(['Precision/Recall with increasing uncertainty | AP ' num2str(AP2) ' | MR ' num2str(MR2)]);
        axis([0, 1, 0, 1]);
        grid on;
        
        outname = [DIR_RES '/' '_ROC_obj_uncertainty' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot);
        % -------------------------
        
        % combined
        % -------------------------
        rocplot = figure();
        hold on;
        plot(vertcat(objPred_sorted.recall), vertcat(objPred_sorted.precision), 'Color', [0.7 0.1 0.2] ); % actual graph
        plot(vertcat(objPred_sorted2.recall), vertcat(objPred_sorted2.precision), 'Color', [0.1 0.2 0.7]); % actual graph
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); % linear decrease
        xlabel('recall'); % x-axis label
        ylabel('precision'); % y-axis label
        %| AP ' num2str(AP) ' | MR ' num2str(MR) '| AP ' num2str(AP2) ' | MR ' num2str(MR2)
        title('Precision/Recall for uncertainty/softmax');
        axis([0, 1, 0, 1]);
        grid on;
        
        outname = [DIR_RES '/' '_ROC_obj_both' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot);
        
        % ###############################################
        % DIFFERENCE GRAPH
        [obj_recall_vals_unique, ia, ~] = unique(vertcat(objPred_sorted.recall));
        obj_prec_vals = vertcat(objPred_sorted.precision);
        obj_prec_vals_unique = obj_prec_vals(ia);
        
        [obj_recall_vals_unique2, ia2, ~] = unique(vertcat(objPred_sorted2.recall));
        obj_prec_vals2 = vertcat(objPred_sorted2.precision);
        obj_prec_vals_unique2 = obj_prec_vals2(ia2);
            
        interp_soft = interp1(obj_recall_vals_unique, obj_prec_vals_unique, interp_x, 'nearest');
        interp_unc = interp1(obj_recall_vals_unique2, obj_prec_vals_unique2, interp_x, 'nearest');
        
        % ------------------------
        %diff_val = mean(prec_vals_interp_clean, 1) - mean(prec_vals_interp_clean2, 1);
        diff_obj_roc = interp_unc - interp_soft;
        AP_diff = trapz(interp_x, diff_obj_roc);
        diffplot = figure();
        hold on;
        %plot(interp_x, diff_val, 'LineWidth', 2); % actual graph
        plot(interp_x, diff_obj_roc, 'LineWidth', 2, 'Color', [0.6 0.2 0.6]); % actual graph
        xlabel('recall') % x-axis label
        ylabel('diff') % y-axis label
        title(['Difference predictive power obj: Softmax vs. uncertainty | AP ' num2str(AP_diff)]);
%         axis([0.0, 1.0, 0.0, 1.0]);
        grid on;

        outname = [NET_OUTPUT '/' '_ROC_obj_diff' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(diffplot);
        % -------------------------
        
        % -------------------------
        % SAVE VARIABLES
        outname = [DIR_RES '/' '_obj_sorted' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS')];
        save(outname, 'objPred_sorted', 'objPred_sorted2', 'diff_obj_roc', 'NET_OUTPUT')
        % -------------------------
    end; clear class ord rocplot;
end;

%% global confusion matrix
% global confusion matrix (after loop)
if((OUT_FIG_CONF || OUT_FIG_ROC))
    if(VERBOSE); fprintf('\ncreating confusion matrix over all data\n'); end
    
    % transpose inputs! (input NxM matrix with N classes, M samples)
    if(OUT_FIG_CONF)
        try
            confusionplot = plotconfusion(summary_groundtruth_oneHot',summary_segmentation_oneHot');
            saveas(confusionplot, [NET_OUTPUT '/' '_CP' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT]);
            close(confusionplot);
        catch ME
            % TODO apparently problems with new dataset?
            fprintf('Apparently problems with confusionmatrix. Not plotted.\n'); 
            disp(ME.identifier);
            disp(ME.message);
        end
    end; clear confusionplot
    if(OUT_FIG_ROC)
        rocplot = plotroc(summary_groundtruth_oneHot',summary_segmentation_oneHot');
        saveas(rocplot, [NAME_RES '_RP' OUT_FIG_FORMAT]);
        close(rocplot)
    end; clear rocplot;
end


if(VERBOSE)
    fprintf('\nDONE\n');
    toc(totalTime);
    fprintf('\n\n\n');
end
if(VERBOSE); fprintf('\n%s\n%s\n', repmat('#',1,70), repmat('#',1,70)); end
