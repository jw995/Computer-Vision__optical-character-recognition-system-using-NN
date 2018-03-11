% Your code here.
clc;
clear;
close all hidden;

file={['../images/01_list.jpg'], ['../images/02_letters.jpg'],...
    ['../images/03_haiku.jpg'], ['../images/04_deep.jpg']};
for i=1:length(file)
    filename=file{i};
    im=imread(filename);
    [lines, bw] = findLetters(im);
end

