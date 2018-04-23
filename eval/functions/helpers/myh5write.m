function myh5write(fname, dsetname, dset)
    chunksize = min(size(dset),20);
   
    % if exist and same extents: overwrite
    try
        h5create(fname, dsetname, dset_size,...
                        'Chunksize', chunksize,...
                        'Datatype', class(dset),...
                        'Deflate', 1);
    catch
    end

    h5write(fname, dsetname, dset);
end