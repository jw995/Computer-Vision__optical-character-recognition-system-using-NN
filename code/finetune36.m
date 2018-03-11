clc;
clear;

num_epoch = 10;
learning_rate = 0.008;

load('../data/nist36_train.mat', 'train_data', 'train_labels')
load('../data/nist36_test.mat', 'test_data', 'test_labels')
load('../data/nist36_valid.mat', 'valid_data', 'valid_labels')
load('../data/nist26_model_60iters.mat')

range=2/sqrt(800);
randm=(rand(10,800)*range-0.5*range);
randb=(rand(10,1)*range-0.5*range);
W{2}=[W{2};randm];
b{2}=[b{2};randb];

loss_list=zeros(1,num_epoch);
accuracy_list=zeros(1,num_epoch);
epoch_list=1:num_epoch;
va_loss=zeros(1,num_epoch);
va_acc=zeros(1,num_epoch);

for j = 1:num_epoch
    
    [W, b] = Train(W, b, train_data, train_labels, learning_rate);

    [train_acc, train_loss] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    [valid_acc, valid_loss] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);
    
    loss_list(1,j)=train_loss;
    accuracy_list(1,j)=train_acc;
    
    va_loss(1,j)= valid_loss;
    va_acc(1,j)=valid_acc;
    

    fprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc, valid_acc, train_loss, valid_loss);
end

plot(epoch_list, loss_list,'r');
hold on;
plot(epoch_list,va_loss,'b');
legend('training data','validation data');
xlabel('epoch');
title('learning rate=0.01, cross-entropy loss');

figure;
plot(epoch_list,accuracy_list,'r'); 
hold on; 
plot(epoch_list,va_acc,'b'); 
legend('training data','validation data');
xlabel('epoch');
title('learning rate=0.01, accuracy')

save('nist36_model_003.mat', 'W', 'b')
