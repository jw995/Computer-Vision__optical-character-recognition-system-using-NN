function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.
[outputs] = Classify(W, b, data);

temp=(labels.*log(outputs));
loss=-sum(temp(:))/(size(temp,1)*size(temp,2));

for i=1:size(data,1)
    [~,idx]=max(outputs(i,:));
    outputs(i,:)=0;
    outputs(i,idx)=1;  
end


total=size(data,1);
err=abs(outputs-labels);
err=sum(err,2);
e=length(find(err==0));
accuracy=e/total;

end
