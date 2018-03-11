clc;
clear;
close all hidden;

% load data
load 'nist36_model_001.mat';
load '../data/nist36_test.mat';

% compute loss and accuracy
[test_acc, test_loss, outputs] = ComputeAccuracyAndLoss(W, b, test_data, test_labels);

layers = [32*32, 800, 36];
[W, b] = InitializeNetwork(layers);
weight=W{1};


for i=1:size(weight,1)
    data=weight(i,:);
    data=reshape(data,32,32);
    grey=mat2gray(data);
    I(:,:,1,i)=grey;
end
montage(I);

[~,hat] = find(outputs==1);
[~,group] = find(test_labels==1);

C = confusionmat(group,hat);
C1=mat2gray(C);
C2=imresize(C1,[260 260],'nearest');

figure;
imshow(C2);



