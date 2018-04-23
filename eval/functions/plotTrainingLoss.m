function res = plotTrainingLoss(files, target_folder, log_folder)
% Creates and saves plot from training logs at target folder 
%
% Logs that are read fulfill are in a certain format.
% Loss from training iterations is handled separately from logs from
% testing iterations.
%
% Example log content:
%I1117 22:46:44.849346  7290 solver.cpp:340] Iteration 7000, Testing net (#0)
%I1117 22:47:18.588232  7290 solver.cpp:408]     Test net output #0: loss = 0.22422 (* 1 = 0.22422 loss)
%I1117 22:47:19.059255  7290 solver.cpp:236] Iteration 7000, loss = 0.125901
%I1117 22:47:19.059288  7290 solver.cpp:252]     Train net output #0: loss = 0.1259 (* 1 = 0.1259 loss)
%I1117 21:23:26.242794  7290 solver.cpp:408]     Test net output #0: loss = 1.24144 (* 1 = 1.24144 loss)
%I1117 21:23:33.131264  7290 solver.cpp:236] Iteration 0, loss = 1.29609
%
% 	INPUT
% 		files			- all log files to concat
% 		target_folder	- puts outfile.png there
% 
% 	OUTPUT
% 		writes out img file
% 
%     AUTHOR: Dominic Mai / Hannes Horneber
% 
%   CHANGELOG:
%         22-Mar-2016 15:00:38:   first version
%         15-Jun-2016 17:16:22:   added 'Position', [x y width height] to
%                                 figure (to fix size)
    
    % save current folder to switch back after saving
	act_folder = pwd;

    % concatenate log text from files
	log_text = [];
	for i = 1:length(files)
		log_file = files{i};
		log_text = [log_text, fileread(log_file)];
    end

	% get training loss from file
	pattern = 'Iteration (\d*), loss = ([\d\.]*(e-)?[\d]*)';    
	[~, tokens_loss] = regexp(log_text, pattern, 'start', 'tokens');

	loss  = zeros(1,numel(tokens_loss));
	iters = zeros(1,numel(tokens_loss));

	for nt = 1:numel(tokens_loss)
		loss(1,nt) = str2double(tokens_loss{nt}{2});
		iters(nt) = str2double(tokens_loss{nt}{1});        
	end

	loss(isnan(loss)) = 0;
    
    % get test loss from file
    % order is swapped (first loss, then Iteration#)
    patternTest = 'Test net output #0: loss = ([\d\.]*(e-)?[\d]*).{1,60}[\n|\r]{1}.{1,60}teration (\d*)';
    [~, tokens_lossTest] = regexp(log_text, patternTest, 'start', 'tokens');

    lossTest  = zeros(1,numel(tokens_lossTest));
	itersTest = zeros(1,numel(tokens_lossTest));
    
    for ntt = 1:numel(tokens_lossTest)
		lossTest(1,ntt) = str2double(tokens_lossTest{ntt}{1});
		itersTest(ntt) = str2double(tokens_lossTest{ntt}{2});        
    end

    % create plot with fixed size
    fig = figure('Visible', 'off', ...
        'PaperPositionMode','auto', ...
        'PaperPosition', [0.1, 0.1, 8, 5]'); % corresponds to 1600x1000
    
    plot(iters, loss);
	hold on;
       
    % plot average loss
	if(length(loss) > 41)
		H = gausswin(41);
		H = H / sum(H);
		avg_loss = conv(loss, H, 'valid');
		n = ceil(length(H) / 2);
		plot(iters(n : end-n+1 ), avg_loss, '-r');
    end
    
    % plot test loss
    plot(itersTest,lossTest, '-g');
        
    % plot settings
    ylabel('loss');
	xlabel('iters');
    % scale axis
	axis([min(iters), max(iters), 0, 2]);
	grid on;
    % add title
	[~, a] = fileparts(pwd);
	title(a, 'Interpreter', 'none');

    % save file
	cd(target_folder);
	outname = ['training_loss_',a,datestr(now,'yyyy-mm-dd-HH-MM-SS'),'.png'];
	print( '-dpng', '-r200', outname);
	close(fig);

    % switch back to current folder
	cd(act_folder);

    % function output
	res.iters = iters;
	res.loss = loss;
	res.avg_loss = avg_loss;
    res.itersTest = itersTest;
    res.lossTest = lossTest;
end
