function [accuracy, loss] =Accuracy_loss(W, b, data, labels, outputs)

% measure the loss 
temp=(labels.*log(outputs));
loss=-sum(temp(:))/(size(temp,1)*size(temp,2));

% compute accuracy
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