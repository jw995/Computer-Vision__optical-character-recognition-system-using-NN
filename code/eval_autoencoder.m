clc;
clear;
close all hidden;

% TODO: load test dataset
load '../data/nist36_test.mat';
load 'net01.mat';

% TODO: reshape and adjust the dimensions to be in the order of [height,width,1,sample_index]
% testdata=reshape(test_data,[32 32 1 size(test_data,1)]);
for i=1:size(test_data,1)
    data=test_data(i,:);
    data1=reshape(data,[32 32]);
    testdata(:,:,1,i)=data1;
end
                      
% TODO: run predict()]
test_recon = double(predict(net,testdata));

for i=1:size(test_data,1)
    temp=test_recon(:,:,1,i);
    result(:,i)=reshape(temp,[1,1024]);
end
result=result';


% show image
i=1723;

[~,label]=max(test_labels,[],2);
correct=label(i,1);
alph={'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z' ...
    '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'};
cla=alph{correct};

figure;
subplot(1,2,1);
origin=reshape(test_data(i,:),[32 32]);
imshow(origin);
title(['label: ',cla]);

subplot(1,2,2);
recon=reshape(result(i,:),[32 32]);
imshow(recon);

peaksnr=0;
for i=1:size(test_data,1)
    origin=test_data(i,:);
    recon=result(i,:);
    peaksnr = peaksnr+psnr(recon,origin);
end
peakavg=peaksnr/size(test_data,1);






