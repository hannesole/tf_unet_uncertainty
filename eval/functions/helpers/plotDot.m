%%
% author: Dominic Mai
function plotDot(i)
%{
	plots a dot. for every 10th a space and for every 100th i and return
%}

	fprintf('.');
	if(mod(i,10) == 0)
		%fprintf('\n');
		fprintf(' ');
	end
	if(mod(i,100) == 0)
		fprintf('%i\n',i);
	end

end
