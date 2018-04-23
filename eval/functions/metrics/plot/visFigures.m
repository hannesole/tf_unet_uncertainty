function [ fig3 ] = visFigures( fig1, fig2 )
%VISFIGURES plots two figures as side by side plots
%   Detailed explanation goes here

    % get handle to axes of figure
    ax1 = gca(fig1);
    ax2 = gca(fig2);

    fig3 = figure; %create new figure
    s1 = subplot(2,1,1); %create and get handle to the subplot axes
    s2 = subplot(2,1,2);

    fig1 = get(ax1,'children'); %get handle to all the children in the figure
    fig2 = get(ax2,'children');

    copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
    copyobj(fig2,s2);
end

