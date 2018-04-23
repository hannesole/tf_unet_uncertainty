function n = linecount(filename, SKIP_EMPTY)
%LINECOUNT counts lines of a textfile
%   tested with files ending with or without newline.
    [fid, msg] = fopen(filename);
    if fid < 0
        error('Failed to open file "%s" because "%s"', filename, msg);
    end
    
    % set optional variables
    if (~exist('SKIP_EMPTY', 'var'))
        SKIP_EMPTY = false;
    end

    n = 0;
    while true
        tline = fgetl(fid);
        if ~ischar(tline)
            break;
        else
            if ~SKIP_EMPTY || ~isempty(strtrim(tline))
                n = n + 1;
            end
        end
    end
    fclose(fid);
end