function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.

W=cell(1,size(layers,2)-1);
b=cell(1,size(layers,2)-1);


for i=1:size(W,2)
    range=2/sqrt(layers(i));
    W{i}=(rand(layers(i),layers(i+1))*range-0.5*range)';
    b{i}=(rand(layers(i),layers(i+1))*range-0.5*range)';
end

end
