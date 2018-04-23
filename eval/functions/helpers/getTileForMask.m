function [ tile_file ] = getTileForMask( mask_file, ...
    TILE_PREFIX, DIR_SRC_tiles, DIR_SRC_tiles_seg3, VERBOSE, APPENDIX )
%GETTILEFORMASK Returns path of a tile based on a mask-name.
% 
% this assumes label-files are in folder DIR_SRC_labels
% and named "tilename copy.tif". It will remove the '8bit_10cm_IRGB_' part
% in the labelname once a TILE_PREFIX is given.
% E.g.:
% label name: 4575000_5266000_8bit_10cm_IRGB_01_01 copy.tif
%   -> tile name: 4575000_5266000_01_01.tif
%
% INPUT:
%
%   mask_file   - the file path to the label
%   APPENDIX    - allows appending something between tilename and fileending
%   TILE_PREFIX - ...?
%   DIR_SRC_tiles           - directory for tiles
%   DIR_SRC_tiles_seg3      - directory for tiles for seg3-labels
%
% OUTPUT:
%
%   tile_file   - the file path to the tile
%
% CHANGELOG: 
%   01-Feb-2017 13:19:40 - first version
%   18-May-2017 14:00:52 - added doc and APPENDIX (for lcc tiles)
% AUTHOR: Hannes Horneber

%% SETTINGS
    % set optional variables
    if (~exist('VERBOSE', 'var'))
        VERBOSE = false;
    end
    if (~exist('DIR_SRC_tiles_seg3', 'var'))
        DIR_SRC_tiles_seg3 = '';
    end
    if (~exist('APPENDIX', 'var'))
        APPENDIX = '';
    end
    
    if(VERBOSE); fprintf('label: %s\n', mask_file); end

%% FUNCTION
    
    % remove ' copy' string
    tmp = regexp(mask_file, ' copy');
    tile_file = [mask_file(1:tmp-1), APPENDIX, '.tif']; 

    % segmentation is based on IRGB
    % => no filenames change needed if IRGB tiles are used
    % otherwise:
    if(~isempty(TILE_PREFIX))
        if(strcmp(TILE_PREFIX,'basic'))
            % seg2: reduce from labelname to basic tilename
            % seg3 isn't altered
            tile_file = strrep(tile_file, '8bit_10cm_IRGB_', '');
        else % allows to insert a TILE_PREFIX
            % seg2: adjust labelname to match specific tilename
            % seg3 isn't altered (stays basic)
            tile_file = strrep(tile_file, '8bit_10cm_IRGB_', TILE_PREFIX);
        end
    end
    if(VERBOSE); fprintf(' %s tile name: %s\n', APPENDIX, tile_file); end

    % distinguished for seg2/seg3 mix
    if(~isempty(DIR_SRC_tiles_seg3))
        if(strfind(mask_file,'8bit_10cm_IRGB_')) 
            tile_file = [DIR_SRC_tiles '/' tile_file];
        else tile_file = [DIR_SRC_tiles_seg3 '/' tile_file];
        end
    else
        tile_file = [DIR_SRC_tiles '/' tile_file];
    end
    
end

