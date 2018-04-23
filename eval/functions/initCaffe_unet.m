function [net, opts] = initCaffe_unet(opts, nChannels)
% 10-May-2016 16:43:19
%{
	for masternet from roberts .prototxt style

	INPUT
        opts		- options array to write into
        nChannels	- number of (image) channels in input

	OUTPUT
        net			- initialized neural network
        opts		- filled options array

    AUTHOR: Dominic Mai, changes by Hannes Horneber

	CHANGELOG:
		previous versions: standard unet, looks for graphicscard and choses ips
        25-Jul-2016 11:33:33 - support images with more than 3 channels

%}
    %% SET PARAMETERS
    % default channel count is 3 (RGB)
	if(~nChannels) nChannels = 3; end
        
	try
		gpu_info = gpuDevice();
	catch
		gpu_info.Name = 'na';
	end
	%...inputsizes
	d4a_size = 1:40;
	input_size  =  (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
	output_size = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;

	%addpath /home/hornebeh/dd/unet-test/caffe-unet/matlab/unet/
	%addpath /home/hornebeh/dd/unet-test/caffe-unet/matlab/
	addpath /misc/lmbraid12/hornebeh/software/caffe-unet/matlab/
    addpath /misc/lmbraid12/hornebeh/software/caffe-unet/matlab/unet
    %addpath /misc/software-lin/lmbsoft/caffe-unet/
    %addpath /misc/software-lin/lmbsoft/caffe-unet/matlab/
    
	caffe.reset_all();
	setenv('HDF5_DISABLE_VERSION_CHECK', '1');

	switch gpu_info.Name
		case 'GeForce GTX TITAN'
			opts.ips = [1,nChannels,540, 540]; 
		case 'GeForce GTX TITAN X'
			opts.ips = [1,nChannels,700, 700]; 
		otherwise
			opts.ips = [1,nChannels,508,508]; %inputsize --> GTX970 4GB
	end

	%% INIT NN CAFFE
	%generate caffe prototxt
	fid = fopen(opts.train_model_def_file);
	trainPrototxt = fread(fid);
	fclose(fid);

	%get layers that are not TRAIN
	fid = fopen(opts.train_model_def_file);
	tline = fgetl(fid);
	stack = {};
	while ischar(tline)
        tline_trimmed = strtrim(tline); % remove leading whitespace
        ind_c1 = regexp(tline_trimmed, '#'); % line begins with #
        
        % ignore lines that are comments (# at the beginning)
        if(~isempty(tline_trimmed) && (isempty(ind_c1) || ind_c1 ~= 1)) % not whole line comment
            % remove comments in line
            ind_c = regexp(tline, '#');
            if(~isempty(ind_c))
                tline = tline(1:(ind_c-1));
            end

            % push criteria: (= include line)
            ind1 = regexp(tline, 'layer', 'once');
            % pop criteria: (= ignore line) 
            % ignore line if TRAIN/TEST layer or comment
            ind2 = regexp(tline, 'TRAIN', 'once');
            ind3 = regexp(tline, 'TEST', 'once');
            
            %push? (if layer)
            if(~isempty(ind1))
                stack{end+1} = tline;
            end
            %pop? (if it was TRAIN/TEST)
            if(~isempty(ind2) || ~isempty(ind3))
                stack(end) = [];
            end
        end
        
		tline = fgetl(fid); % next line
	end
	fclose(fid);
	pat = regexp(stack{1}, 'bottom:[ ]+''[\w]+''', 'match');

	%write out tmp-def
	model_def_file = 'tmp-def.prototxt';
	fid = fopen(model_def_file,'w');
	ips = opts.ips;
	fprintf(fid, 'input: "data"\n'); 
	fprintf(fid, 'input_shape: {dim: %i dim: %i dim: %i dim: %i}\n', ips(1), ips(2), ips(3), ips(4)); 
	fprintf(fid, 'state: { phase: TEST }\n'); 

	%change bottom most input to data
	s = sprintf('%s', trainPrototxt);
	s = strrep(s, pat{1}, 'bottom: ''data''');

	fwrite(fid, s);
	fclose(fid);

	caffe.set_mode_gpu();
	if( isfield(opts, 'gpu_device'))
	  caffe.set_device(opts.gpu_device)
	end

	%% Initialize a network
	net = caffe.Net(model_def_file, opts.model_file, 'test');

end
