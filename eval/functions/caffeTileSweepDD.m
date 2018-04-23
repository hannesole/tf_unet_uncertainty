function valid_map = caffeTileSweepDD(net, data, border_pad, ips, outputMode, multiSweep)
% Di 18. Aug 10:21:55 CEST 2015
%{
	perform tiling on 2D data with unet, probably 4up (scan all 90°).

	!!!Caffe needs to be initialized already!!!

	INPUT
		data		-	normalized image data
		border_pad	-	how much border is "lost" when passing through the NN
		ips			-	inputsize of the NN
        outputMode  -   specifies type of output (= raw output layer or
                        computed scoreMap)
		multiSweep	-	pass 4 times with 90° rotation and average?


	OUTPUT
		valid scoremap, same size as data

    AUTHOR: Dominic Mai, changes made by Hannes Horneber

	CHANGELOG:
		11-Nov-2015: new caffe version with new matlab interface
        17-May-2016 14:47:42: parameter displayMode
        06-Jun-2016 17:19:29: additional displayModes
%}

	verbose = 1;
	rotations = [0];
	if(nargin == 5)
		multiSweep = 0;
	end

	if(multiSweep ~= 0)
		rotations = [0,1,2,3];
	end

	map_acc = zeros([size(data,1), size(data,2)]);
	c = 1;
	for k = rotations

		data_rot = rot90(data,k);

		%add mirror border to data
		data_padded = padarray(data_rot, [border_pad, border_pad,0], 'symmetric', 'both');
		step_size = ips(end) - 2*border_pad;

		shape = size(data_rot);
		shape = shape(1:2);

		p = ceil(shape / step_size) * step_size - shape;
		data_padded = padarray(data_padded, [p,0], 'post');
		tmp = ceil(shape / step_size);

		s = size(data_padded);
		scoremap = zeros(s(1:2));

		if(k==0 && verbose)
			fprintf('\n(x,y,rotations) %i * %i * %i = %i Tiles \n', tmp(1), tmp(2), length(rotations), tmp(1)*tmp(2)*length(rotations));
		end

		for i = 0:tmp(1)-1
			for j = 0:tmp(2)-1

				r1 = int32(i*step_size + 1:((i+1)*step_size + 2*border_pad));
				r2 = int32(j*step_size + 1:((j+1)*step_size + 2*border_pad));

				tile = data_padded(r1,r2,:);
				scores_caffe = net.forward( {tile});
                
                switch outputMode
                    case 0
                        % adjust background values by addition (of negative class values)
                        scores_caffe{1}(:,:,1) = ...
                            scores_caffe{1}(:,:,1) + scores_caffe{1}(:,:,2) + scores_caffe{1}(:,:,3);
                        
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                    case 1
                        % get one output layer only
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = scores_caffe{1}(:,:,1);
                    case 2
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = scores_caffe{1}(:,:,2);
                    case 3
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = scores_caffe{1}(:,:,3);
                    case 4
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                    case 5
                        % adjust background values by addition (of negative class values)
                        scores_caffe{1}(:,:,1) = ...
                            scores_caffe{1}(:,:,1) + scores_caffe{1}(:,:,2) + scores_caffe{1}(:,:,3);
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                    case 6
                        % adjust background values by subtracting a fixed
                        % value from background 
                        % (5.5 evaluated by student descent on tile 7)
                        scores_caffe{1}(:,:,1) = ...
                            scores_caffe{1}(:,:,1) - 5.5;
                        
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                    case 7
                        % adjust background values by fixed multiplier 
                        % (0.3 evaluated by student descent on tile 7)
                        scores_caffe{1}(:,:,1) = ...
                            scores_caffe{1}(:,:,1) * 0.3;
                        
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                                   scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                    case 8
                        % get maximum
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = max(scores_caffe{1}, [], 3);
                    case 9
                        % get minimum
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = min(scores_caffe{1}, [], 3);
                    case 10
                        % depreacated - normalization is local and varies
                        % from tile to tile (doesn't generalize)
                        
                        % normalize all three output layers
                        for dim_i = 1:3
                            layer = scores_caffe{1}(:,:,dim_i);
                            layer = layer - min(layer(:));
                            layer = layer / max(layer(:));
                            scores_caffe{1}(:,:,dim_i) = layer;
                        end
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                                   scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                    case 11
                        % get one output layer only
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = scores_caffe{1}(:,:,1);
                    case 12
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = scores_caffe{1}(:,:,2);
                    case 13
                        scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = scores_caffe{1}(:,:,3);
                    otherwise
                        % segmentation
                        % get output_layer with max activiation (= class)
                        [~, class] = max(scores_caffe{1}, [], 3);
                                   scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) ...
                            = class;
                end


                % segmentation
                %[val, ind] = max(scores_caffe{1}, [], 3);
                %scoremap(r1(border_pad+1:end-border_pad), r2(border_pad+1:end-border_pad)) = ind;
                
                % demonstration of segmentation
                %[val, ind] = max(scores_caffe{1}, [], 3);
                %whos ind
                %figure, imagesc(ind)
                
                if(verbose)
					plotDot(c);	
				end
				c = c+1;
			end
		end

		valid_map = scoremap(border_pad+1:end-border_pad-p(1), border_pad+1:end-border_pad-p(2));

		map_acc = map_acc + rot90(valid_map, -k);
    end
    
	valid_map = map_acc ./ length(rotations);
end
