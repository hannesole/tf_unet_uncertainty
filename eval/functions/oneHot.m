function oneHotLabels = oneHot(labels, labelValues)
%ONEHOT one-hot encoding for elements of label vector
%{
    Takes a vector of size n by 1 as input and 
    creates a one-hot encoding of its elements.

	AUTHOR: Nicolas
            (minor modification by Hannes Horneber, 27-Sep-2016 16:58:33)
            https://de.mathworks.com/matlabcentral/fileexchange/35364-fast-
            multilayer-feedforward-neural-network-training/content/oneHot.m
%}

    if (~exist('labelValues', 'var'))
        labelValues = unique(labels);
    end
    nLabels = length(labelValues);
    nSamples = length(labels);

    oneHotLabels = zeros(nSamples, nLabels);

    for i = 1:nLabels
        oneHotLabels(:,i) = (labels == labelValues(i));
    end
end