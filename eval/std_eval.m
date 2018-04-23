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


%% Initialize
LIB_DIR = 'D:\Hannes Horneber\Documents\Wissen\Studium\Computer Science\WS17-18 STD\code\projects\win_tf_unet\eval\functions'

% addpath functions
% addpath functions/helpers
addpath (genpath(LIB_DIR));
% addpath functions/metrics
totalTime = tic;

%% Settings I

% dir with net output: tile, groundtruth, label, softmax, (uncertainty)
NET_OUTPUT = 'data_net/ex4_trainset';
% specify directory to write outputs of script
DIR_RES = [ NET_OUTPUT '/rerun' datestr(now,'yyyy-mm-dd-HH-MM-SS') ]; 

% ##############################>   OUTPUT OPTIONS    <####################
% OUTPUT OPTIONS (choose output: toggle true/false to plot or not)
OUT_TILE =          false;      % tile itself
OUT_OVL =           true;      % tile with segmentation overlay
OUT_SEG =           false;      % segmentation itself
OUT_IOU =           false;      % [labeled data] IoU map
OUT_OBJ =           true;      % [labeled data] object based stats
OUT_OBJ_PLOT =      false;      % [labeled data] plot of iou_map with obj
OUT_PIX_ROC =       true;      % [labeled data] pixelwise ROC
OUT_TRUTH =         false;      % [labeled data] groundtruth
OUT_IRGB =          false;      % false-color (replaces red with NIR)
OUT_DSM =           false;      % height model image
OUT_FIG_CONF =      true;      % [labeled data] figure with confusion matrix
OUT_FIG_CONF_PI =   false;      % [labeled data] ... per image
OUT_FIG_ROC =       false;      % [labeled data] figure with ROC
OUT_OUTPUT_LAYER =  false;      % pseudo-probability map for each class

N_QUANTILES =       20;
MAX_SCORE =         1.0;

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
fprintf('List raw tifs from %s\n-a.\n', NET_OUTPUT);
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
graphs_uncertainty = zeros([data_n, N_QUANTILES, 2]);
graphs_softmax = zeros([data_n, N_QUANTILES, 2]);
quantiles = 1:N_QUANTILES
quantiles = quantiles * MAX_SCORE / N_QUANTILES


%%
% iterate through data: img_i = 1:length(img_idx) -> img_idx(img_i)
for i = 1:data_n
    % ##########################> LOAD DATA <##############################
    %tile, label, pred, softmax 
    load(data{i})
    if(VERBOSE); fprintf('Loaded tile_%i\n', i); end
    tile = double(tile);
    label = boolean(label); %uint8(label);
    softmax = double(softmax);
    pred = boolean(pred); %uint8(pred);
    uncertainty(:,:,2) = double(uncertainty(:,:));
    
    [~, tile_name, ~] = fileparts(data{i});
    tile_name = [DIR_RES '/' tile_name];
    % ##########################> PRINT TILE <#############################
    % print original tile (RGB and/or IRGB)
    if(OUT_TILE)
        % write tile without georef
        imwrite(tile(:,:,[1 2 3]), [tile_name '-a' OUT_TILE_FORMAT]);
    end

    % #########################> PRINT OUTPUT <############################
    % >>>>> PROBABILITY PER CLASS CLASS OUTPUT <<<<<<<
    % needed for object score
    % can be used to print the activations per class (before max)

    % get output layer for each class
    output_layers = softmax;

    if(OUT_OUTPUT_LAYER)
        % scale from 0 - 1 (softmax)
        ctop = 1.0;
        cbottom = 0.0;
        % output mode single layers fixed scale
        for i_layer = 1:NOUTCLASSES
            if(VERBOSE); fprintf('  > softmax class %i\n', i_layer); end
            % write result graphics
            f = figure('visible', 'off', 'OuterPosition', [0, 0, 900, 900]);
            a = axes;
            colormap jet;
            image(softmax(:,:,i_layer), 'cdatamapping', 'scaled')
            % output mode single layers fixed scale
            title(['Tile ' int2str(i) ': output layer ' int2str(i_layer-1)]);
            % shared color scale
            caxis manual;
            caxis([cbottom ctop]);          
            colorbar('peer', a);
            saveas(f, [tile_name '-z' num2str(i_layer) OUT_FIG_FORMAT]);
            close(f);
        end; clear i_layer;
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
            if(VERBOSE); fprintf('  > pixel IoU tile %i, class %i: %f\n', i, desired_class, iou(row,desired_class+1)); end  
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
        [objTruth, objTruth_map] = im2obj(groundtruth_classmask); 
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
        
        % assign scores to objects
        for i=1:length(objPred)   
            objPred(i).score = mean(softmax_map(objPred_map == i));
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
    
graphs_uncertainty = zeros([data_n, N_QUANTILES, 2]);
graphs_softmax = zeros([data_n, N_QUANTILES, 2]);
quantiles = 1:N_QUANTILES
quantiles = quantiles * MAX_SCORE / N_QUANTILES

    % PIXELWISE PRECISION/RECALL
    if(OUT_PIX_ROC)
%         figure(); imshow(label);
%         figure(); imshow(pred);
%         figure(); imagesc(uncertainty(:,:,1))


        for g = 1:2
            if g==1; score = uncertainty(:,:,1); 
            else score = 1 - softmax(:,:,2); end
            max_score = 1.0; %max(uncertainty(:));
            q_steps = 15;

            % allocate graph
            graph(q_steps) = struct('quantile',[],'precision',[],'recall',[]);
            % create graph
            for q = 1:q_steps
                graph(q).quantile = max_score * (q/q_steps);
%                 fprintf('showing quantile %.2f\n', graph(q).quantile);
                % select pixels of quantile
                label_q = label;
                label_q(score > graph(q).quantile) = false;
                pred_q = pred;
                pred_q(score > graph(q).quantile) = false;
                %imshow(label_q); waitforbuttonpress;

                % true/false positives (precision) and false negatives (recall)
                tp = label_q;
                tp(~pred_q) = false;

                fn = label; % calc recall on all pixels, not just quantile
                fn(pred_q) = false;

                fp = pred_q;
                fp(label_q) = false;

                % compute prec/recall
                graph(q).precision = nnz(tp) / (nnz(tp) + nnz(fp));
                graph(q).recall = nnz(tp) / (nnz(tp) + nnz(fn));
%                 fprintf('precision %.3f / recall %.3f\n', graph(q).precision, graph(q).recall);
            end
            graphs(i, g) = graph;
        end; clear graph
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
end


%% global pixelwise ROC out
if(OUT_PIX_ROC)
    for g = 1:2
    
    end
end

%% global obj out

if(OUT_OBJ)
    iou_thresh = 0.1
    for class = 1
        objPredClass = objPred_all{class};
        objTruthClass = objTruth_all{class};
        % allow sorted access, start with highest score
        [~,ord] = sort([objPredClass.score], 'descend');
        objPred_sorted = objPredClass(ord);

        for i=1:length(objPred_sorted)
            objPred_sorted(i).recall = length(find(vertcat(objPred_sorted(1:i).iou) > iou_thresh)) / ...
                length(objTruthClass);
            objPred_sorted(i).precision = length(find(vertcat(objPred_sorted(1:i).iou) > iou_thresh)) / i;
        end; clear i objPredClass objTruthClass;

        save('obj.mat','objPred_all','objTruth_all')

        % calc area under curve
        AP = trapz(vertcat(objPred_sorted.recall), vertcat(objPred_sorted.precision));

        rocplot = figure();
        hold on;
        plot(vertcat(objPred_sorted.recall), vertcat(objPred_sorted.precision));
        line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
        xlabel('recall') % x-axis label
        ylabel('precision') % y-axis label
        title(['ROC with AP ' num2str(AP)]);
        axis([0, 1, 0, 1]);
        grid on;

        outname = [NET_OUTPUT '/' '_ROC' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT];
        print( '-dpng', '-r200', outname);
        close(rocplot);
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
            saveas(confusionplot, [NET_OUTPUT '/' '_cp' '_' datestr(now,'dd-mm-yyyy-HH-MM-SS') OUT_FIG_FORMAT]);
        catch ME
            % TODO apparently problems with new dataset?
            fprintf('Apparently problems with confusionmatrix. Not plotted.\n'); 
            disp(ME.identifier);
            disp(ME.message);
        end
    end; clear confusionplot;
    if(OUT_FIG_ROC)
        rocplot = plotroc(summary_groundtruth_oneHot',summary_segmentation_oneHot');
        saveas(rocplot, [NAME_RES '_rp' OUT_FIG_FORMAT]);
    end; clear rocplot;
end

if(VERBOSE)
    fprintf('\nDONE\n');
    toc(totalTime);
    fprintf('\n\n\n');
end
if(VERBOSE); fprintf('\n%s\n%s\n', repmat('#',1,70), repmat('#',1,70)); end
