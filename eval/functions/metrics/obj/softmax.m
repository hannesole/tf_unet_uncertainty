function [ softmax ] = softmax( net_output )
%SOFTMAX softmax of convolutional neural net output
  
    softmax = zeros(size(net_output));
    %softmax = exp(net_output) ./ sum(exp(net_output), 3);
    
    % layerwise computation
    for class = 1:size(net_output, 3)
        softmax(:,:,class) = exp(net_output(:,:,class)) ./ sum(exp(net_output), 3);
    end
end

