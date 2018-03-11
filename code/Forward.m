function [out, act_h, act_a] = Forward(W, b, X)
% [OUT, act_h, act_a] = Forward(W, b, X) performs forward propogation on the
% input data 'X' uisng the network defined by weights and biases 'W' and 'b'
% (as generated by InitializeNetwork(..)).
%
% This function should return the final softmax output layer activations in OUT,
% as well as the hidden layer post activations in 'act_h', and the hidden layer
% pre activations in 'act_a'.

act_a=cell(1,size(W,2)-1,1);
act_h=cell(1,size(W,2)-1,1);

% input layer and hidden layer
for i=1:size(act_a,2)
    weight=W{i};
    hidden_size=size(weight,1);
    act_a{i}=zeros(hidden_size,1);
    
    for j=1:hidden_size
        if (i==1)
            act_a{i}(j,1)=sum(weight(j,:).*X);
            if (strcmp('true',isnan(act_a{i}(j,1))))
                act_a{i}(j,1)=0;
            end
        else
            act_a{i}(j,1)=sum(weight(j,:)'.*act_h{i-1});
             if (strcmp('true',isnan(act_a{i}(j,1))))
                act_a{i}(j,1)=0;
            end
        end
    end
    
    act_h{i}=1./(1+exp(-act_a{i}));
end

% calculate output layer
weight=W{i+1};
hidden_size=size(weight,1);
pre_out=zeros(hidden_size,1);

for j=1:hidden_size
    pre_out(j,1)=sum(weight(j,:)'.*act_h{i});
end

exp_pre=exp(pre_out);
out=exp_pre/sum(exp_pre);

    

end
