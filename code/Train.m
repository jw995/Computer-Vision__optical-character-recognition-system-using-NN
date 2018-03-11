function [W, b, loss, accuracy] = Train(W, b, train_data, train_label, learning_rate)
% [W, b] = Train(W, b, train_data, train_label, learning_rate) trains the network
% for one epoch on the input training data 'train_data' and 'train_label'. This
% function should returned the updated network parameters 'W' and 'b' after
% performing backprop on every data sample.


% This loop template simply prints the loop status in a non-verbose way.
% Feel free to use it or discard it

outputs=zeros(size(train_label));

p = randperm(size(train_data,1));

 
for j = 1:size(p,2)
    i=p(j);
    
    X=train_data(i,:);
    Y=train_label(i,:)';
    
    % get output layer 
    [out, act_h, act_a] = Forward(W, b, X);

    % update weight
    [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a);
    [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate);

    
end

end
