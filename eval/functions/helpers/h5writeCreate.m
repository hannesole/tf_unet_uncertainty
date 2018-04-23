% creates h5 container and writes dset into it
% APPENDABLE - true: makes dset_size Inf, so that contents can be appended
%
% 03-Feb-2017 13:03:05 - Hannes Horneber
function h5writeCreate(filename, datasetname, dataset, APPENDABLE)
    % set optional variables
    if (~exist('APPENDABLE', 'var'))
        APPENDABLE = true; % default is overwrite
    end
    
    % h5 params
    dset_size = size(dataset);
    chunksize = [dset_size(1:end-1) 1];
    %min(dset_size, 20);
    
    if(APPENDABLE) 
        h5create(filename, datasetname, [dset_size(1:end-1) Inf],...
                    'Chunksize', chunksize,...
                    'Datatype', class(dataset),...
                    'Deflate', 1); % appendable (Inf)
                
        % write in append mode
        h5write(filename, datasetname, dataset, ... 
            ones(1, length(dset_size)), dset_size); 
        
    else % if exist and same extents: overwrite
        h5create(filename, datasetname, dset_size,...
                    'Chunksize', chunksize,...
                    'Datatype', class(dataset),...
                    'Deflate', 1);
        % write first contents
        h5write(filename, datasetname, dataset);
    end    
end