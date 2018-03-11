clc;
clear;
close all hidden;

% TODO: load training dataset
load '../data/nist36_train.mat';


% TODO: reshape and adjust the dimensions to be in the order of [height,width,1,sample_index]
finedata=reshape(train_data,[32 32 1 size(train_data,1)]);
finelabel=zeros(1,1,64,size(train_data,1));

for i=1:size(train_data,1)
    data=train_data(i,:);
    label=imresize(data,[8 8]);
    finelabel(1,1,:,i)=reshape(label,[1 1 64]);
end

layers = define_autoencoder();

options = trainingOptions('sgdm',...
                          'InitialLearnRate',0.001,...
                          'MaxEpochs',3,...
                          'MiniBatchSize',20,...
                          'Shuffle','every-epoch',...
                          'Plot','training-progress',...
                          'VerboseFrequency',20);

% TODO: run trainNetwork()
net = trainNetwork(finedata,finedata,layers,options);
save net01;

                      