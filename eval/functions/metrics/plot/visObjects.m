function [ fig ] = visObjects( img, obj, labelfield, displaytext_arg, SUBPLOT)
%VISOBJECTS DEPRECATED creates image with colorized connected components
%   https://de.mathworks.com/help/images/ref/label2rgb.html
    
    % displayed labels for the objects
    if strcmp(labelfield, '@composite')
        displaytext = strcat(...
            cellfun(@num2str, {obj.id}, 'UniformOutput', false), ...
            repmat({':['},1,length(obj)),...
            cellfun(@num2str, {obj.overlaps}, 'UniformOutput', false), ...
            repmat({'->'},1,length(obj)), ...
            cellfun(@num2str, {obj.iou}, 'UniformOutput', false), ...
            repmat({']'},1,length(obj)));
    elseif strcmp(labelfield, '@arg')
        displaytext = displaytext_arg;
    else
        displaytext = cellfun(@num2str, {obj.(labelfield)}, 'UniformOutput', false);
    end
    
    if(~exist('SUBPLOT','var')); SUBPLOT = false; end
    
    % numbered regions
    % figure, imshow(label2rgb(labels, @colorcube, [0.5 0.5 0.5])); hold on;
    fig = figure('Name','visObjects');
    if(SUBPLOT); subplot(1,2,2); end;
    imshow(img); hold on;
    colors=['b' 'g' 'r' 'c' 'm' 'y'];
    for i=1:length(obj)
        % color: use colors(cidx) to alternate colors
%          cidx = mod(i,length(colors))+1;
        
        % plot boundary
%         boundary = obj(j).boundary;
%         plot(boundary(:,2), boundary(:,1), 'Color', [1 0 0] , 'LineWidth', 2); 

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
        
        % plot text (boundary)
%         %  randomize text position for better visibility
%         rndRow = ceil(length(boundary)/(mod(rand*j,7)+1));
%         col = boundary(rndRow,2); row = boundary(rndRow,1);       
%         h = text(col+1, row-1, num2str(j));
%         set(h,'Color',colors(cidx),'FontSize',12);
    end; clear i col row rndRow h;
    hold off;
end

