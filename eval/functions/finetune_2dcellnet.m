function cmd = finetune_2dcellnet(opts)
% Di 2. Aug 12:08:47 CEST 2016
%{
	This script converts labelled training data such that it can be used
	for finetuning a 2d cellnet to this data.
	Data needs to be in [train_data_folder] as pairs of .tif files:

		filename_xxx.tif		-	gray valued (i.e. 1channel) image data
		filename_xxx.labels.tif	-	corresponding label file as indexed image.
	
	Labelling conventions for filename_xxx.labels.tif:
	0	-	background
	1	-	ignore: use this for image regions(pixels) that you want to ignore.
					This is useful for example when
					-you don't want to label all objects in your image
					-for regions that are ambiguous
					-regions that are out of focus 
	>=2 -	instancelabels for fg objects / cells: Same label can be used for multiple cells,
			however touching cells must have pairwise different instancelabels.



	This script creates a run_caffe.sh shell script in a folder [NETNAME] below your [base_folder]
	You have to execute this script to perform the actual fine-tuning.

	If you also provide a test set, run evaluatue_iou_2dcellnet.m to automatically evaluate the fine-tuned models.
%}

	d4a_size = 8:60;
	input_size  =  (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
	output_size = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;

	%snap inputsize to nearest 'good' inputsize
	[~, ind] = min(abs(input_size - opts.net_size_x));
	opts.net_size_x = input_size(ind);
	[~, ind] = min(abs(input_size - opts.net_size_y));
	opts.net_size_y = input_size(ind);

	opts
	addpath(opts.base_folder);


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%parameters for weight generation - don't change unlesss you know what you're doing
	wanted_element_size_um = 0.5;

	sigma1_um = 5;
	sigma1_px = sigma1_um / wanted_element_size_um;
	foregroundBackgroundRatio = 0.1;

	borderWeightFactor = 50;
	borderWeightSigma_um = 3; 
	sigma2_px= borderWeightSigma_um / wanted_element_size_um;
	scale = opts.el_size_um / wanted_element_size_um;
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% 1. read files, create training data
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	cd(opts.base_folder);
	mkdir(opts.NETNAME);

	cd(opts.train_data_folder);
	tmp = dir('*.labels.tif');
	files = {tmp.name};
	%absolute path
	for f_nr = 1:length(files)
		files{f_nr} = [pwd,'/',files{f_nr}];
	end

	tfiles = {};

	fprintf('Preparing training data...\n');

	area_stats = [];
	for fi = 1:length(files)
		
		fname = files{fi};
		fprintf('Processing %s (%i of %i)\n', fname, fi, length(files));
		instancelabels = imread(fname);

		if strcmp(opts.force_unique_labelling, 'true')
			
			labels = unique(instancelabels(:));
			labels(labels==0) = []; %bg
			labels(labels==1) = []; %ignore

			instancelabels_unique = instancelabels;
			counter = 2;
			for li = 1:length(labels)
				
				mask = instancelabels == labels(li);
				cc = bwconncomp(mask);
				plist = cc.PixelIdxList;
                % label connected components starting from 2
				for pii = 1:length(plist)
					
					instancelabels_unique(plist{pii}) = counter;
					counter = counter + 1;
				end
			end

			instancelabels = instancelabels_unique;
		end

		data_file = strrep(fname, '.labels', '');
		data = single(imread(data_file));

		%rescale?
		if scale ~= 1
			fprintf('rescaling data by %0.1f\n', scale);
			data = imresize(data, scale, 'bicubic');
			instancelabels = imresize(instancelabels, scale, 'nearest');
		end

		labels = unique(instancelabels(:));
		labels(labels==0) = [];
		labels(labels==1) = [];

		areas = [];
		for li = 1:length(labels)
			canvas = instancelabels==labels(li);
			areas(end+1) = sum(canvas(:));
		end
		area_stats(fi).areas = areas;

		data = data - min(data(:));
		data = data / max(data(:));

		ignore_mask = instancelabels == 1; 

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%create "extra weights" based on instancelabels
		labels = zeros(size(instancelabels));
		extraweights = zeros(size(instancelabels));
		extraweights2 = zeros(size(instancelabels));

		sDisk = strel('disk',1);
		bordermask = zeros(size(instancelabels));

		inds = unique(instancelabels(:));
		inds(inds == 0) = []; %remove bg
		inds(inds == 1) = []; %remove ignore 

		for li = 1:length(inds)
			mask = (instancelabels== inds(li));
			bordermask = bordermask + (imdilate(mask,sDisk));%touching borders > 1
		end

		mask2 = single(ismember(instancelabels, inds));
		mask2(bordermask>1) = 0;

		labels = mask2;

		min1dist = 1e10*ones(size(instancelabels)); % --> dists for gaussian border weight decay
		min2dist = 1e10*ones(size(instancelabels)); % --> dists for touching
		for li = 1:length(inds)

			mask = (instancelabels==inds(li));
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

		weights = zeros(size(instancelabels),'single');
		weights(labels > 0) = 1;

		weights(labels == 0) = extraweights(labels == 0) + extraweights2(labels == 0) + foregroundBackgroundRatio;
		weights(ignore_mask) = 0;
		
		weights2 = zeros(size(instancelabels),'single');
		weights2(labels > 0) = 1;
		weights2(labels == 0) = foregroundBackgroundRatio;
		weights2(ignore_mask) = 0;
		%End  weight extravaganza
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		%write out
		outname = [strrep(data_file, '.tif', '.h5')];

		[~,~] = system(['rm ', outname]);
		myh5write2(outname, '/data', single(data), wanted_element_size_um, [size(data),1]);
		myh5write2(outname, '/weights', weights, wanted_element_size_um, [size(data),1]);
		myh5write2(outname, '/weights2', weights2, wanted_element_size_um, [size(data),1]);
		myh5write2(outname, '/instancelabels', uint16(instancelabels), wanted_element_size_um, [size(data),1]);
		myh5write2(outname, '/labels', labels, wanted_element_size_um, [size(data),1]);
		tfiles{fi} = outname;
	end

	fig = figure('Visible', 'off');
	a = [area_stats.areas];
	hist(a);
	xlabel('Area of scale normalized cells in pixel');
	ylabel('Count');

	a = sort(a);
	ind = ceil(0.05*length(a));
	thresh = a(ind);
	title(sprintf('0.95 acceptance rate of trainset at %i pixels', thresh));

	outname = [opts.base_folder, '/', opts.NETNAME, '/area_stats.png'];
	fprintf('Writing out %s\n', outname);
	print( '-dpng', '-r200', outname);
	close(fig);



	fprintf('Creating configuration files for caffe training for: %s\n', opts.NETNAME);
	cd(opts.base_folder)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% 2. create config for finetuning
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%generate filelist 
	outname = [opts.NETNAME, '/input_files.txt'];

	fid = fopen(outname,'w');
	s =[];
	for i = 1:length(tfiles)
		s = [s, sprintf('%s\n', tfiles{i})];
	end

	fwrite(fid, s);
	fclose(fid);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%net definition 
	outname = [opts.NETNAME, '/net_def.prototxt'];

	fid = fopen('templates/net_def_template.prototxt');
	trainPrototxt = fread(fid);
	fclose(fid);
	fid = fopen(outname,'w');
	s = sprintf('%s', trainPrototxt);
	s = strrep(s, '@@@netname@@@', opts.NETNAME);
	s = strrep(s, '@@@net_size_x@@@', sprintf('%i', opts.net_size_x));
	s = strrep(s, '@@@net_size_y@@@', sprintf('%i', opts.net_size_y));

	fwrite(fid, s);
	fclose(fid);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%solver conf
	outname = [opts.NETNAME, '/solver_conf.prototxt'];

	fid = fopen('templates/solver_conf_template.prototxt');
	trainPrototxt = fread(fid);
	fclose(fid);
	fid = fopen(outname,'w');
	s = sprintf('%s', trainPrototxt);
	s = strrep(s, '@@@max_iter@@@', sprintf('%i', opts.num_its));
	s = strrep(s, '@@@snapshot@@@', sprintf('%i', opts.snapshot));
	s = strrep(s, '@@@solver_mode@@@', opts.caffe_mode);

	fwrite(fid, s);
	fclose(fid);


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%script
	outname = [opts.NETNAME, '/run_caffe.sh'];

	%FS
	fid = fopen('templates/run_caffe_template_ft.sh');
	trainPrototxt = fread(fid);
	fclose(fid);
	fid = fopen(outname,'w');
	s = sprintf('%s', trainPrototxt);
	s = strrep(s, '@@@netname@@@', [pwd, '/', opts.NETNAME]);
	s = strrep(s, '@@@target_folder@@@', [pwd, '/', opts.NETNAME]);
	s = strrep(s, '@@@baseline_model@@@', [opts.baseline_model]);
	fwrite(fid, s);
	fclose(fid);
	system(['chmod +x ', opts.NETNAME, '/run_caffe.sh']);

	fprintf('Preparation done. Please execute "run_caffe.sh" in the folder\n %s\n', [pwd, '/', opts.NETNAME]);

	cmd = [pwd, '/', opts.NETNAME, '/run_caffe.sh'];
end

