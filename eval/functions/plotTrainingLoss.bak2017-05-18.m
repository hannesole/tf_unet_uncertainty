function res = plotTrainingLoss(files, target_folder)
% Di 22. Mär 15:00:38 CET 2016
%{
	INPUT
		files			- all log files to concat
		target_folder	- puts outfile.png there

	OUTPUT
		writes out img file

    AUTHOR: Dominic Mai, changes by Hannes Horneber

    CHANGELOG:
        15-Jun-2016 17:16:22:   added 'Position', [x y width height] to
                                figure (to fix size)
                                - doesn't work
        25-Nov-2016 16:51:33:   
%}

	act_folder = pwd;

	log_text = [];
	for i = 1:length(files)
		log_file = files{i};
		log_text = [log_text, fileread(log_file)];
	end


	%get actual loss from file
	pattern = 'Iteration (\d*), loss = ([\d\.]*(e-)?[\d]*)';    
	[starts_loss, tokens_loss] = regexp(log_text, pattern, 'start', 'tokens');

	loss = zeros(1, numel(tokens_loss));
	iters = zeros(1,numel(tokens_loss));

	for nt = 1:numel(tokens_loss)
		loss(1,nt) = str2double(tokens_loss{nt}{2});
		iters(nt) = str2double(tokens_loss{nt}{1});        
	end


	loss(isnan(loss)) = 0;

	fig = figure('Visible', 'off', 'Position', [0 0 1200 900]);
	%fig = figure;
	plot(iters, loss);
	hold on;

	if(length(loss) > 41)
		H = gausswin(41);
		H = H / sum(H);
		avg_loss = conv(loss, H, 'valid');
		n = ceil(length(H) / 2);
		plot(iters(n : end-n+1 ), avg_loss, '-r');
	end
	ylabel('loss');
	xlabel('iters');

	axis([min(iters), max(iters), 0, 2]);
	grid on;
	[~, a] = fileparts(pwd);
	title(a, 'Interpreter', 'none');

	cd(target_folder);
	outname = ['training_loss_',a,datestr(now,'dd-mm-yyyy-HH-MM-SS'),'.png'];
	print( '-dpng', '-r200', outname);
	close(fig);

	cd(act_folder);

	res.iters = iters;
	res.loss = loss;
	res.avg_loss = avg_loss;
end
