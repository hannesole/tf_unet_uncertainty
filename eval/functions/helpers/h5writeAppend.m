% appends data to existing h5 container
% h5 container needs to be appendable (size Inf)
%
% 03-Feb-2017 13:03:05 - Hannes Horneber
function h5writeAppend(filename, datasetname, dataset)
    info = h5info(filename, datasetname); % get info on dataset to append to
    curSize = info.Dataspace.Size; % determine start
    h5write(filename, datasetname, dataset, ...
             [ones(1, length(curSize)-1) curSize(end)+1], size(dataset));
end