function [ fig ] = plotROC( recall, precision, fig )
%PLOTROC plot receiver operator characteristic in figure
%   if no figure_handle is passed, a figure will be created

    if(~exist('fig','var'))
        fig = figure();
    else figure(fig); % make fig current figure handle
    end;
    
    % average precision
    AP = trapz(recall, precision);
    
    hold on;
    plot(recall, precision);
    line([0 1], [1 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
    xlabel('recall') % x-axis label
    ylabel('precision') % y-axis label
    title(['ROC with AP ' num2str(AP)]);
    axis([0, 1, 0, 1]);
    grid on;
end