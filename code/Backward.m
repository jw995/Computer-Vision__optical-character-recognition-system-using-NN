function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_h' and 'act_a' the network layer pre and post activations when forward
%     forward propogating the input smaple 'X'

% calculate output layer
last_layer=act_h{size(act_h,2)};
weight=W{size(W,2)};
hidden_size=size(weight,1);
pre_out=zeros(hidden_size,1);

for j=1:hidden_size
    pre_out(j,1)=sum(weight(j,:)'.*last_layer);
end
exp_pre=exp(pre_out);
out=exp_pre/sum(exp_pre);

% compute loss
loss=-(Y.*log(out));  % for only one term


% initialize gradient cell
grad_W=cell(size(W));
grad_b=cell(size(b));

% compute gradient
% 1- from loss to output --------------------------
loss_out=out-Y;
out_last=last_layer;

grad_W{size(W,2)}=loss_out*out_last';
                             
grad_b{size(W,2)}=loss_out;

                             
% 2- hidden layers --------------------------------
for j=1:size(W,2)-1 
    i=size(W,2)-j;
    % get the sum of each column of previous weight gradient matrix
    pre_grad=grad_W{i+1};
    pre=pre_grad/(out_last');
    
    hidden=act_h{i};
    sigm=hidden.*(1-hidden);
    
    if (i==1)
        pre_act=X;
    else
        pre_act=act_h{i-1}';
    end
    
    grad_W{i}=(pre'*W{i+1})'.*sigm*pre_act;
    grad_b{i}=(pre'*W{i+1})'.*sigm;
end

end
