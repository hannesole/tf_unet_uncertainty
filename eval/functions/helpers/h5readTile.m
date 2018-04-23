% appends data to existing h5 container
% h5 container needs to be appendable (size Inf)
%
% 03-Feb-2017 13:03:05 - Hannes Horneber
function tile = h5readTile(filename, datasetname, tileIdx)
    info = h5info(filename, datasetname); % get info on dataset to append to
    curSize = info.Dataspace.Size; % determine start
%     fprintf('read in %s\n %s \n to %s\n', ...
%         filename, ...
%         mat2str([ones(1, length(curSize)-1) tileIdx]), ...
%         mat2str([curSize(1:end-1) tileIdx])); 
    tile = h5read(filename, datasetname, ...
             [ones(1, length(curSize)-1) tileIdx], [curSize(1:end-1) 1]);
end