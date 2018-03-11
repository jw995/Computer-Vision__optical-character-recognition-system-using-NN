function [text] = extractImageText(fname)
% [text] = extractImageText(fname) loads the image specified by the path 'fname'
% and returns the next contained in the image as a string.
im=imread(fname);
[lines, bw] = findLetters(im);
load 'nist36_model_89.mat';

alph=['A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z' ...
    '0' '1' '2' '3' '4' '5' '6' '7' '8' '9']';

for i=1: length(lines)
    ch=double(lines{i});
    ch = imresize(ch,[32 32]);
    ch=ch/max(ch(:));
    ch=reshape(ch,1,1024);
    
    [out, ~,~] = Forward(W, b, ch);
    [~,idx]=max(out(:,1));
    text(1,i)=alph(idx,1);
end

end
