function [ fig ] = plotObjects( img, obj, label, fig, varargin)
%PLOTOBJECTS plot image with colorized connected components in figure
%   
%   INPUT:
%       img     -   background image
%       obj     -   struct with objects
%       label   -   specify label that is displayed for each object
%                    either a cellarray (length of obj, one label per obj(i)
%                    or a string: 
%                       '@composite' (generates composite labels: id/overlaps/iou)
%                       'structfield' (generate label from obj.structfield)
%                       '' (generate default labels 1:length(obj))
%       fig     -   figure handle to plot in
%                    if no figure_handle is passed, a figure will be created
%
%   OUTPUT:
%       fig     -   figure handle of plot
    
%% SETTINGS
opts = struct( ...                      %# define defaults
    'BOUNDARY', false, ...      % show boundaries
    'BBOX', true ...            % show bounding boxes
);
%# override options defaults if propertyName/propertyValue pairs are given
optionNames = fieldnames(opts);         %# read acceptable option names
nArgs = length(varargin);               %# count arguments
if round(nArgs/2)~=nArgs/2; error('Function needs propertyName/propertyValue pairs'); end
for pair = reshape(varargin,2,[])       %# pair is {propName;propValue}
   inpName = upper(pair{1});            %# make case insensitive
   if any(strcmp(inpName,optionNames)); opts.(inpName) = pair{2};
   else error('%s is not a recognized parameter name',inpName); end
end; clear inpName pair;

%% SETTINGS II
    % set labels that are displayed for the objects
    if(iscell(label))
        % if string array is passed, then 
        displaytext = label;
    else
        if isempty(label);
            displaytext = cellfun(@num2str, num2cell(1:length(obj)), 'UniformOutput', false);
        elseif strcmp(label, '@composite')
            displaytext = strcat(...
                cellfun(@num2str, {obj.id}, 'UniformOutput', false), ...
                repmat({':['},1,length(obj)),...
                cellfun(@num2str, {obj.overlaps}, 'UniformOutput', false), ...
                repmat({'->'},1,length(obj)), ...
                cellfun(@num2str, {obj.iou}, 'UniformOutput', false), ...
                repmat({']'},1,length(obj)));
        else
            displaytext = cellfun(@num2str, {obj.(label)}, 'UniformOutput', false);
        end
    end
    
    if(~exist('fig','var'))
        fig = figure();
    else
        try 
            figure(fig); % make fig current figure handle
        catch
            fig = figure();
        end
    end;
    
%% ACTUAL PLOT

    % numbered regions
    % figure, imshow(label2rgb(labels, @colorcube, [0.5 0.5 0.5])); hold on;
    imshow(img); hold on;
    colors=['b' 'g' 'r' 'c' 'm' 'y'];
    for i=1:length(obj)
        % color: use colors(cidx) to alternate colors
        cidx = mod(i,length(colors))+1;
        
        if opts.BBOX
            % plot bounding boxes
            bb = obj(i).BoundingBox;
            rectangle('Position', bb,...
                'EdgeColor', [0.7 0.7 0.7], 'LineWidth', 1)

            % plot text
            col = bb(1) ; row = bb(2);
            h = text(col+3, row+3, displaytext{i});

            % use smaller font for longer text
            if(max(cellfun(@length, displaytext) > 4))
                %set(h,'Color',colors(cidx),'FontSize',8);
                set(h,'Color',[0.3 0.3 0.9],'FontSize',8);
            else set(h,'Color',[0.3 0.3 0.9],'FontSize',10,'FontWeight','bold');
            end
        end
        
       if opts.BOUNDARY
            % change bbox color to match boundary
            set(h,'Color',colors(cidx));

            % plot boundary
            boundary = obj(i).boundary;
            plot(boundary(:,2), boundary(:,1), 'Color', colors(cidx), 'LineWidth', 2);

            %randomize text position for better visibility
            col = boundary(5,2); row = boundary(5,1);
            if opts.BBOX
            	h = text(col+1, row-1, num2str(i));
            else
                h = text(col+1, row+1, displaytext{i});
            end
            set(h,'Color',colors(cidx),'FontSize',12);
        end
        
        % plot text (boundary)
%         %  randomize text position for better visibility
%         rndRow = ceil(length(boundary)/(mod(rand*j,7)+1));
%         col = boundary(rndRow,2); row = boundary(rndRow,1);       
%         h = text(col+1, row-1, num2str(j));
%         set(h,'Color',colors(cidx),'FontSize',12);
    end; clear i col row rndRow h;
    hold off;
end

